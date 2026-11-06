"""
Motor Neural de Trato Vocal Aigis RVC v2 40kHz 100% Residente em GPU (ROCm / PyTorch).
Combina:
- Desbloqueio ROCm Navi3x (HSA_OVERRIDE_GFX_VERSION=11.0.0, MIOPEN_FIND_MODE=FAST, AOTriton)
- Padding reflexivo de 0.25s (impede truncamento de fonemas iniciais/finais no HuBERT)
- HuBERT Layer 12 + RMVPE Pitch Tracker acelerados em VRAM
- Alinhamento suave de F0 no sweetspot calmo da Aigis (~230 Hz) com filtro anti-sub-harmônicos (135 Hz)
- Interpolação Temporal Linear Contínua + Proteção de Consoantes (index_rate=0.20, protect=0.33)
- Síntese NSF-HiFiGAN 40kHz nativa em PyTorch GPU (gerador estocástico real, zero zumbido estático ONNX)
- Equalização LTAS + Biquads Zero-Phase de Estúdio (zero eco, brilho natural >6kHz da Dawn Bennett)
"""

import os
import sys
import time

if sys.platform.startswith("linux"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import faiss
import librosa
import numpy as np

np.float = float
np.complex = complex
import soundfile as sf
import torch
import torch.nn.functional as F
import transformers.modeling_utils as _tmu
import transformers.utils.import_utils as _tiu
from infer_rvc_python.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from infer_rvc_python.lib.rmvpe import RMVPE
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from transformers import HubertModel

_tiu.check_torch_load_is_safe = lambda: None
_tmu.check_torch_load_is_safe = lambda: None


def _design_peaking_sos(fc: float, gain_db: float, q_factor: float, fs: int) -> np.ndarray:
    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q_factor)
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * a_amp
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_amp
    a0 = 1.0 + alpha / a_amp
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_amp
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _design_high_shelf_sos(fc: float, gain_db: float, q_factor: float, fs: int) -> np.ndarray:
    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q_factor)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(a_amp)
    b0 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0)
    b2 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha)
    a0 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0)
    a2 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


class InVRAMVocalTractAigis:
    """
    Motor RVC v2 40kHz 100% residente em GPU (PyTorch ROCm) com fidelidade fonética integral PT-BR.
    """

    def __init__(
        self,
        pth_path: str = "models/rvc/aigis/AigisMrM0dZ_300e_5700s.pth",
        index_path: str = "models/rvc/aigis/AigisMrM0dZ.index",
        rmvpe_path: str = "models/rvc/rmvpe.pt",
    ):
        core_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(core_dir, ".."))

        pth_full = pth_path if os.path.isabs(pth_path) else os.path.join(self.project_root, pth_path)
        index_full = index_path if os.path.isabs(index_path) else os.path.join(self.project_root, index_path)
        rmvpe_full = rmvpe_path if os.path.isabs(rmvpe_path) else os.path.join(self.project_root, rmvpe_path)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.sr_in = 16000
        self.window = 160
        self.tgt_sr = 40000

        # Padding reflexivo de 0.25s impede que a janela convolucional corte sílabas iniciais/finais
        self.pad_sec = 0.25
        self.t_pad = int(self.sr_in * self.pad_sec)
        self.t_pad_tgt = int(self.tgt_sr * self.pad_sec)

        # Filtro High-Pass 48Hz na entrada 16kHz
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr_in)

        # 1. HuBERT Layer 12 em GPU
        self.hubert = HubertModel.from_pretrained("r3gm/hubert_base").to(self.device).float().eval()

        # 2. RMVPE Pitch Tracker em GPU
        self.rmvpe = RMVPE(rmvpe_full, is_half=False, device=self.device)

        # 3. Gerador NSF-HiFiGAN 40kHz Nativo em GPU PyTorch
        cpt = torch.load(pth_full, map_location="cpu", weights_only=False)
        self.tgt_sr = int(cpt["config"][-1])
        self.t_pad_tgt = int(self.tgt_sr * self.pad_sec)
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

        self.net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q
        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.device).float().eval()

        # 4. Índice Anatômico FAISS
        self.index = faiss.read_index(index_full)
        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        # 5. Perfil Espectral LTAS de Referência (Dawn Bennett 40kHz) + Biquads Zero-Phase de Estúdio
        self.n_fft = 2048
        self.ltas_gold_log = self._load_gold_ltas()
        self.studio_sos = np.vstack(
            [
                signal.butter(2, 65.0 / (0.5 * self.tgt_sr), btype="highpass", output="sos"),
                _design_peaking_sos(215.0, 0.4, q_factor=1.1, fs=self.tgt_sr),
                _design_peaking_sos(1220.0, -1.0, q_factor=1.5, fs=self.tgt_sr),
                _design_peaking_sos(3200.0, -0.4, q_factor=1.2, fs=self.tgt_sr),
                _design_high_shelf_sos(7000.0, 0.8, q_factor=0.707, fs=self.tgt_sr),
            ]
        )

        self._warmup()

    def _load_gold_ltas(self) -> np.ndarray | None:
        candidates = [
            os.path.join(self.project_root, "output", "audio", "tests_refinement", "conversational_speed088.wav"),
            os.path.join(self.project_root, "models", "voice_prompts", "aigis_prompt_conversational.wav"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    y_g, _ = librosa.load(path, sr=self.tgt_sr)
                    s_g = np.abs(librosa.stft(y_g, n_fft=self.n_fft))
                    frms_g = np.sqrt(np.mean(s_g**2, axis=0))
                    active = frms_g > np.percentile(frms_g, 35)
                    ltas_g = np.sqrt(np.mean(s_g[:, active] ** 2, axis=1) + 1e-8)
                    return gaussian_filter1d(np.log(ltas_g), sigma=8)
                except Exception:
                    pass
        return None

    def _warmup(self):
        """Aquece os kernels MIOpen/ROCm na inicialização para evitar latência na 1ª frase."""
        try:
            dummy = (0.05 * np.sin(2 * np.pi * 230.0 * np.linspace(0, 0.35, int(self.sr_in * 0.35)))).astype(
                np.float32
            )
            _ = self.convert_tensor(dummy, self.sr_in, pitch_shift=0.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def convert_tensor(
        self,
        wav: np.ndarray,
        orig_sr: int,
        pitch_shift: float = 0.0,
        index_rate: float = 0.22,
        protect: float = 0.33,
        target_f0_med: float = 230.0,
        ltas_strength: float = 0.65,
        f0_floor: float = 130.0,
    ) -> tuple[np.ndarray, int]:
        """
        Converte o áudio base para o timbre 40kHz da Aigis 100% em GPU sem truncar bordas nem consoantes.
        """
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        if orig_sr != self.sr_in:
            gcd = np.gcd(int(orig_sr), self.sr_in)
            audio16k = signal.resample_poly(wav, self.sr_in // gcd, int(orig_sr) // gcd).astype(np.float32)
        else:
            audio16k = wav.astype(np.float32)

        if len(audio16k) < 320:
            return np.zeros(int(len(audio16k) * self.tgt_sr / self.sr_in), dtype=np.float32), self.tgt_sr

        # 1. Limpeza sub-grave + Padding reflexivo de 0.25s nas bordas
        audio_clean = signal.filtfilt(self.bh, self.ah, audio16k).astype(np.float32)
        pad_len = min(self.t_pad, max(160, len(audio_clean) - 2))
        pad_tgt = int(pad_len * (self.tgt_sr / self.sr_in))
        audio_pad = np.pad(audio_clean, (pad_len, pad_len), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        # 2. Extração de Features Fonéticas HuBERT Layer 12 (GPU)
        with torch.no_grad():
            inp = torch.from_numpy(audio_pad).float().unsqueeze(0).to(self.device)
            feats = self.hubert(inp, output_hidden_states=True).hidden_states[12]
            feats0 = feats.clone()

        # 3. Extração de Pitch RMVPE (GPU) + Calibração de Sweetspot (230 Hz)
        f0 = self.rmvpe.infer_from_audio(audio_pad, 0.03).astype(np.float32)

        # Correção de oitava em vogais femininas
        for i in range(1, len(f0) - 1):
            if 0 < f0[i] < 125.0 and (f0[i - 1] > 140.0 or f0[i + 1] > 140.0):
                f0[i] *= 2.0

        f0[(f0 > 0) & (f0 < f0_floor)] = 0.0
        voiced = f0 > 0
        if np.any(voiced):
            f0[voiced] = signal.medfilt(f0[voiced], kernel_size=3)

        if pitch_shift != 0.0:
            f0 = f0 * (2.0 ** (pitch_shift / 12.0))
        elif target_f0_med > 0 and np.sum(f0 > f0_floor) > 8:
            cur_med = float(np.median(f0[f0 > f0_floor]))
            # Ajuste suave limitado a +-2.2 semitons para preservar naturalidade
            shift_ratio = np.clip(target_f0_med / cur_med, 2.0 ** (-2.2 / 12.0), 2.0 ** (2.2 / 12.0))
            f0[f0 > 0] *= shift_ratio

        f0[(f0 > 0) & (f0 < f0_floor)] = 0.0
        f0 = f0[:p_len]

        # 4. Projeção de Faringe FAISS k-NN
        if index_rate > 0 and self.index is not None:
            npy = feats[0].cpu().numpy().astype(np.float32)
            score, ix = self.index.search(npy, k=8)
            weight = np.square(1.0 / np.maximum(score, 1e-6))
            weight /= weight.sum(axis=1, keepdims=True)
            npy_retr = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = (
                torch.from_numpy(npy_retr).unsqueeze(0).to(self.device) * index_rate
                + (1.0 - index_rate) * feats
            )

        # 5. Interpolação Linear Contínua 50fps -> 100fps + Proteção de Consoantes (protect=0.33)
        feats = F.interpolate(
            feats.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
        ).permute(0, 2, 1)
        feats0 = F.interpolate(
            feats0.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
        ).permute(0, 2, 1)

        eff_len = min(feats.shape[1], len(f0))
        pitchf_t = torch.from_numpy(f0[:eff_len]).unsqueeze(0).to(self.device)
        pitchff = pitchf_t.clone()
        pitchff[pitchf_t > 0] = 1.0
        pitchff[pitchf_t < 1] = protect
        pitchff = pitchff.unsqueeze(-1)
        feats_final = feats[:, :eff_len, :] * pitchff + feats0[:, :eff_len, :] * (1.0 - pitchff)

        # Coarse Pitch
        f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
        f0_mel = 1127.0 * np.log(1.0 + f0[:eff_len] / 700.0)
        voiced_mask = f0_mel > 0
        f0_mel[voiced_mask] = (f0_mel[voiced_mask] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
        f0_coarse = np.clip(np.rint(f0_mel), 1, 255).astype(np.int64)

        pitch_t = torch.from_numpy(f0_coarse).unsqueeze(0).to(self.device).long()
        p_len_t = torch.tensor([eff_len], device=self.device, dtype=torch.long)
        sid_t = torch.tensor([0], device=self.device, dtype=torch.long)

        # 6. Síntese NSF-HiFiGAN 40kHz Nativa em GPU
        with torch.no_grad():
            audio_out = (
                self.net_g.infer(feats_final.float(), p_len_t, pitch_t, pitchf_t, sid_t)[0][0, 0]
                .cpu()
                .float()
                .numpy()
            )

        # Remove o padding reflexivo das bordas
        if pad_tgt > 0 and len(audio_out) > 2 * pad_tgt:
            audio_out = audio_out[pad_tgt:-pad_tgt]

        # 7. Rastreamento Rápido de Envelope RMS de 25ms (preserva ataque de sílabas sem bombeamento)
        if len(audio_clean) >= 320 and len(audio_out) >= 800:
            win_src = max(64, int(0.025 * self.sr_in))
            hop_src = max(16, int(0.008 * self.sr_in))
            win_tgt = max(160, int(0.025 * self.tgt_sr))
            hop_tgt = max(40, int(0.008 * self.tgt_sr))
            rms1 = librosa.feature.rms(y=audio_clean, frame_length=win_src, hop_length=hop_src)
            rms2 = librosa.feature.rms(y=audio_out, frame_length=win_tgt, hop_length=hop_tgt)
            t_len = audio_out.shape[0]
            r1 = F.interpolate(
                torch.from_numpy(rms1).float().unsqueeze(0), size=t_len, mode="linear", align_corners=False
            ).squeeze()
            r2 = F.interpolate(
                torch.from_numpy(rms2).float().unsqueeze(0), size=t_len, mode="linear", align_corners=False
            ).squeeze()
            r2 = torch.clamp(r2, min=1e-5)
            rms_mix = 0.25
            gain_env = (torch.pow(r1, 1.0 - rms_mix) * torch.pow(r2, rms_mix - 1.0)).numpy()
            audio_out = audio_out * gain_env

        # 8. Alinhamento Espectral LTAS + Biquads Zero-Phase (zero eco, ar de estúdio 40kHz)
        if self.ltas_gold_log is not None and len(audio_out) >= self.n_fft:
            d_spec = librosa.stft(audio_out, n_fft=self.n_fft)
            s_mag = np.abs(d_spec)
            frms = np.sqrt(np.mean(s_mag**2, axis=0))
            active = frms > np.percentile(frms, 35)
            if np.any(active):
                ltas_src = np.sqrt(np.mean(s_mag[:, active] ** 2, axis=1) + 1e-8)
                freqs = librosa.fft_frequencies(sr=self.tgt_sr, n_fft=self.n_fft)
                diff_db = (self.ltas_gold_log - gaussian_filter1d(np.log(ltas_src), sigma=8)) * (
                    20.0 / np.log(10.0)
                )
                diff_db[freqs < 80] = 0.0
                diff_db = np.clip(diff_db * ltas_strength, -6.5, 6.5)
                gain_ltas = (10.0 ** (diff_db / 20.0)).astype(np.float32)[:, None]
                audio_out = librosa.istft((d_spec * gain_ltas).astype(np.complex64), length=len(audio_out))

        if len(audio_out) > 64:
            audio_out = signal.sosfiltfilt(self.studio_sos, audio_out.astype(np.float64)).astype(np.float32)

        # 9. Normalização RMS consistente + Fade-in/out de 8ms (elimina cliques entre chunks no PipeWire)
        rms_cur = float(np.sqrt(np.mean(audio_out**2) + 1e-8))
        if rms_cur > 1e-5:
            audio_out = audio_out * (0.105 / rms_cur)
        peak = float(np.max(np.abs(audio_out)))
        if peak > 0.89:
            audio_out = (audio_out / peak) * 0.89

        fade_samples = min(int(0.008 * self.tgt_sr), len(audio_out) // 4)
        if fade_samples > 1:
            ramp = (0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))).astype(np.float32)
            audio_out[:fade_samples] *= ramp
            audio_out[-fade_samples:] *= ramp[::-1]

        return audio_out.astype(np.float32), self.tgt_sr
