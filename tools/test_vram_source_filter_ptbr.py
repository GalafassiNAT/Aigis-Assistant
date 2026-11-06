"""
Etapa 1 — Protótipo Auditivo & Benchmark de Latência In-VRAM (40.000 Hz)
Unifica:
1. Fonte Fonética/Prosódica Nativa PT-BR (Kokoro StyleTTS2 Local com Álgebra de Vetores Emocionais vs Edge-TTS Francisca)
2. Trato Vocal / Faringe Aigis RVC v2 40kHz 100% em VRAM (PyTorch ROCm / CUDA) com os 4 pilares anatômicos restaurados:
   - Dicionário de Faringe FAISS (AigisMrM0dZ.index, k=8, index_rate=0.75)
   - Proteção de Consoantes Surdas e Respiração (protect=0.33)
   - Interpolação Temporal Linear Suave + Padding Reflexivo (sem degraus de 50Hz e sem cliques de borda)
   - Filtro High-Pass Butterworth 48Hz + Estabilização Dinâmica de Envelope RMS (rms_mix_rate=0.25)
"""

import asyncio
import io
import os
import re
import sys
import time
from pathlib import Path

# Otimizações de ambiente antes de inicializar PyTorch/ROCm
if sys.platform.startswith("linux"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import edge_tts
import faiss
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from infer_rvc_python.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from infer_rvc_python.lib.rmvpe import RMVPE
from kokoro_onnx import Kokoro
from scipy import signal
from transformers import HubertModel


def normalize_pt_phonetics(text: str) -> str:
    return re.sub(r"\bAigis\b", "Áiguis", text, flags=re.IGNORECASE)


def match_rms_envelope(
    src_audio: np.ndarray,
    src_sr: int,
    tgt_audio: np.ndarray,
    tgt_sr: int,
    rms_mix_rate: float = 0.25,
) -> np.ndarray:
    """Ancora o envelope dinâmico de energia à fala original para eliminar oscilações/tremores."""
    rms1 = librosa.feature.rms(
        y=src_audio, frame_length=src_sr // 2 * 2, hop_length=src_sr // 2
    )
    rms2 = librosa.feature.rms(
        y=tgt_audio, frame_length=tgt_sr // 2 * 2, hop_length=tgt_sr // 2
    )
    t_len = tgt_audio.shape[0]
    r1 = F.interpolate(
        torch.from_numpy(rms1).float().unsqueeze(0), size=t_len, mode="linear", align_corners=False
    ).squeeze()
    r2 = F.interpolate(
        torch.from_numpy(rms2).float().unsqueeze(0), size=t_len, mode="linear", align_corners=False
    ).squeeze()
    r2 = torch.clamp(r2, min=1e-6)
    gain = (torch.pow(r1, 1.0 - rms_mix_rate) * torch.pow(r2, rms_mix_rate - 1.0)).numpy()
    return tgt_audio * gain


class InVRAMVocalTractAigis:
    """Motor RVC v2 40kHz 100% residente em VRAM com modelo anatômico completo de faringe."""

    def __init__(
        self,
        pth_path: str = "models/rvc/aigis/AigisMrM0dZ_300e_5700s.pth",
        index_path: str = "models/rvc/aigis/AigisMrM0dZ.index",
        rmvpe_path: str = "models/rvc/rmvpe.pt",
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.sr_in = 16000
        self.window = 160
        self.tgt_sr = 40000
        # Padding reflexivo de 0.30s evita artefatos de borda na atenção do HuBERT e convoluções HiFiGAN
        self.pad_sec = 0.30
        self.t_pad = int(self.sr_in * self.pad_sec)
        self.t_pad_tgt = int(self.tgt_sr * self.pad_sec)

        # Filtro Butterworth High-Pass 48Hz (remove DC offset e rumble sub-grave)
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr_in)

        t0 = time.perf_counter()
        print(f"[*] Alocando Trato Vocal Aigis 40kHz em {self.device}...")

        self.hubert = HubertModel.from_pretrained("r3gm/hubert_base").to(self.device).float().eval()
        self.rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)

        cpt = torch.load(pth_path, map_location="cpu")
        self.tgt_sr = cpt["config"][-1]
        self.t_pad_tgt = int(self.tgt_sr * self.pad_sec)
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

        self.net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q
        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.device).float().eval()

        self.index = faiss.read_index(index_path)
        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        print(
            f"[+] Trato Vocal pronto em {(time.perf_counter() - t0)*1000:.0f}ms "
            f"(SR={self.tgt_sr}Hz | Vetores de Faringe FAISS={self.index.ntotal})"
        )

    def convert_tensor(
        self,
        wav: np.ndarray,
        orig_sr: int,
        pitch_shift: float = 0.0,
        index_rate: float = 0.75,
        protect: float = 0.33,
        rms_mix_rate: float = 0.25,
        f0_median_filter: int = 3,
    ) -> tuple[np.ndarray, int, dict]:
        t_start = time.perf_counter()

        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        if orig_sr != self.sr_in:
            gcd = np.gcd(int(orig_sr), self.sr_in)
            audio16k = signal.resample_poly(wav, self.sr_in // gcd, int(orig_sr) // gcd).astype(
                np.float32
            )
        else:
            audio16k = wav.astype(np.float32)

        # 1. Limpeza sub-grave + Padding reflexivo nas bordas
        audio_clean = signal.filtfilt(self.bh, self.ah, audio16k).astype(np.float32)
        audio_pad = np.pad(audio_clean, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        # 2. Extração Fonética Contínua (HuBERT Layer 12)
        t_h0 = time.perf_counter()
        with torch.no_grad():
            inp = torch.from_numpy(audio_pad).float().unsqueeze(0).to(self.device)
            feats = self.hubert(inp, output_hidden_states=True).hidden_states[12]
            feats0 = feats.clone()  # Cópia pura para proteção de consoantes surdas
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_hubert = (time.perf_counter() - t_h0) * 1000

        # 3. Rastreamento Glotal F0 (RMVPE + Filtro Mediano Anti-Tremor)
        t_r0 = time.perf_counter()
        f0 = self.rmvpe.infer_from_audio(audio_pad, 0.03)
        if f0_median_filter >= 3:
            # Suaviza apenas regiões vozeadas sem borrar transições surdas
            voiced = f0 > 0
            f0_filt = signal.medfilt(f0, kernel_size=f0_median_filter)
            f0[voiced] = f0_filt[voiced]
        if pitch_shift != 0.0:
            f0 = f0 * (2.0 ** (pitch_shift / 12.0))

        f0 = f0[:p_len]
        f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        voiced_mask = f0_mel > 0
        f0_mel[voiced_mask] = (f0_mel[voiced_mask] - f0_mel_min) * 254.0 / (
            f0_mel_max - f0_mel_min
        ) + 1.0
        f0_coarse = np.clip(np.rint(f0_mel), 1, 255).astype(np.int64)
        dt_rmvpe = (time.perf_counter() - t_r0) * 1000

        # 4. Projeção de Faringe via FAISS k-NN + Interpolação Linear + Proteção de Consoantes
        t_f0 = time.perf_counter()
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

        # Interpolação LINEAR suave (elimina o degrau quadrado de 50Hz do nearest-neighbor!)
        feats = F.interpolate(
            feats.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
        ).permute(0, 2, 1)
        feats0 = F.interpolate(
            feats0.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
        ).permute(0, 2, 1)

        eff_len = min(feats.shape[1], len(f0_coarse))
        pitch_t = torch.from_numpy(f0_coarse[:eff_len]).unsqueeze(0).to(self.device).long()
        pitchf_t = torch.from_numpy(f0[:eff_len].astype(np.float32)).unsqueeze(0).to(self.device)

        if protect < 0.5:
            pitchff = pitchf_t.clone()
            pitchff[pitchf_t > 0] = 1.0
            pitchff[pitchf_t < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats[:, :eff_len, :] * pitchff + feats0[:, :eff_len, :] * (1.0 - pitchff)
        else:
            feats = feats[:, :eff_len, :]
        dt_faiss = (time.perf_counter() - t_f0) * 1000

        # 5. Síntese Neural NSF-HiFiGAN 40kHz
        t_g0 = time.perf_counter()
        p_len_t = torch.tensor([eff_len], device=self.device, dtype=torch.long)
        sid_t = torch.tensor([0], device=self.device, dtype=torch.long)
        with torch.no_grad():
            audio_out = (
                self.net_g.infer(feats.float(), p_len_t, pitch_t, pitchf_t, sid_t)[0][0, 0]
                .cpu()
                .float()
                .numpy()
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_netg = (time.perf_counter() - t_g0) * 1000

        # Remove o padding reflexivo das pontas
        if self.t_pad_tgt > 0 and len(audio_out) > 2 * self.t_pad_tgt:
            audio_out = audio_out[self.t_pad_tgt : -self.t_pad_tgt]

        # 6. Estabilização de Envelope RMS + Normalização Segura (-1.0 dBFS = 0.89)
        if rms_mix_rate < 1.0:
            audio_out = match_rms_envelope(
                audio_clean, self.sr_in, audio_out, self.tgt_sr, rms_mix_rate=rms_mix_rate
            )

        peak = np.max(np.abs(audio_out))
        if peak > 0:
            audio_out = (audio_out / peak) * 0.89

        dt_total = (time.perf_counter() - t_start) * 1000
        metrics = {
            "hubert_ms": dt_hubert,
            "rmvpe_ms": dt_rmvpe,
            "faiss_ms": dt_faiss,
            "netg_ms": dt_netg,
            "rvc_total_ms": dt_total,
        }
        return audio_out.astype(np.float32), self.tgt_sr, metrics


async def synth_francisca_memory(text: str) -> tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    comm = edge_tts.Communicate(text, "pt-BR-FranciscaNeural", rate="-4%", pitch="-1Hz")
    mp3_bytes = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            mp3_bytes.extend(chunk["data"])
    wav, sr = sf.read(io.BytesIO(bytes(mp3_bytes)), dtype="float32")
    dt = (time.perf_counter() - t0) * 1000
    return wav, sr, dt


async def main():
    out_dir = Path("output/audio/tests_etapa1_40khz")
    out_dir.mkdir(parents=True, exist_ok=True)

    kokoro = Kokoro("models/kokoro/kokoro-v1.0.onnx", "models/kokoro/voices-v1.0.bin")
    tract = InVRAMVocalTractAigis()

    # Vetores de estilo base do Kokoro (510, 1, 256)
    s_dora = kokoro.get_voice_style("pf_dora")
    s_heart = kokoro.get_voice_style("af_heart")
    s_bella = kokoro.get_voice_style("af_bella")
    s_sky = kokoro.get_voice_style("af_sky")

    # Warmup completo na GPU
    print("\n[*] Executando warmup dos kernels ROCm/MIOpen...")
    w_wav, w_sr = kokoro.create("Olá mestre.", voice=s_dora, speed=1.0, lang="pt-br")
    _ = tract.convert_tensor(w_wav, w_sr)
    print("[+] Warmup concluído!\n")

    phrase_std = normalize_pt_phonetics(
        "Boa noite, mestre! Eu sou a Aigis. Todos os meus sistemas estão operacionais e prontos para a sua ordem."
    )
    phrase_mel = normalize_pt_phonetics(
        "O fato de existir um futuro não garante que todos terão acesso a ele... É algo tão óbvio, e ainda assim, demoramos a compreender."
    )
    phrase_tac = normalize_pt_phonetics(
        "Entrando em modo de combate! Miras alinhadas e sistemas de defesa ativos. Aguardando instruções!"
    )
    phrase_short = normalize_pt_phonetics(
        "Entendido, mestre! Iniciando varredura agora."
    )

    kokoro_cases = [
        (
            "01_kokoro_dora_puro_40khz",
            phrase_std,
            s_dora,
            0.96,
            0.0,
        ),
        (
            "02_kokoro_blend_conversational_40khz",
            phrase_std,
            0.70 * s_dora + 0.30 * s_heart,
            0.96,
            0.0,
        ),
        (
            "03_kokoro_blend_melancholic_40khz",
            phrase_mel,
            0.65 * s_dora + 0.35 * s_bella,
            0.88,
            -0.8,
        ),
        (
            "04_kokoro_blend_tactical_40khz",
            phrase_tac,
            0.75 * s_dora + 0.25 * s_sky,
            1.02,
            0.5,
        ),
        (
            "05_clausula_curta_tempo_real_40khz",
            phrase_short,
            0.70 * s_dora + 0.30 * s_heart,
            0.96,
            0.0,
        ),
    ]

    print("=" * 78)
    print("🚀 BENCHMARK ETAPA 1 — PIPELINE UNIFICADA IN-VRAM @ 40.000 Hz")
    print("=" * 78)

    for name, text, style_vec, speed, p_shift in kokoro_cases:
        t0 = time.perf_counter()
        base_wav, base_sr = kokoro.create(text, voice=style_vec, speed=speed, lang="pt-br")
        dt_tts = (time.perf_counter() - t0) * 1000

        out_wav, out_sr, m = tract.convert_tensor(base_wav, base_sr, pitch_shift=p_shift)
        total_ms = dt_tts + m["rvc_total_ms"]
        dur_s = len(out_wav) / out_sr
        rtf = (total_ms / 1000.0) / dur_s

        out_path = out_dir / f"{name}.wav"
        sf.write(str(out_path), out_wav, out_sr)
        print(
            f"\n✅ {name}.wav ({dur_s:.2f}s @ {out_sr}Hz)\n"
            f"   ├─ Fonte PT-BR (Kokoro Local): {dt_tts:6.1f} ms\n"
            f"   ├─ Trato Vocal RVC 40kHz GPU : {m['rvc_total_ms']:6.1f} ms "
            f"[HuBERT={m['hubert_ms']:.0f}ms | RMVPE={m['rmvpe_ms']:.0f}ms | FAISS={m['faiss_ms']:.0f}ms | NetG={m['netg_ms']:.0f}ms]\n"
            f"   └─ TOTAL FIM-A-FIM           : {total_ms:6.1f} ms (RTF = {rtf:.3f}x)"
        )

    # Comparativo com Francisca Neural em memória + o mesmo Trato Vocal restaurado
    base_fr, sr_fr, dt_fr = await synth_francisca_memory(phrase_std)
    out_fr, out_sr_fr, m_fr = tract.convert_tensor(base_fr, sr_fr, pitch_shift=0.0)
    tot_fr = dt_fr + m_fr["rvc_total_ms"]
    dur_fr = len(out_fr) / out_sr_fr
    out_fr_path = out_dir / "06_francisca_neural_restaurada_40khz.wav"
    sf.write(str(out_fr_path), out_fr, out_sr_fr)
    print(
        f"\n✅ 06_francisca_neural_restaurada_40khz.wav ({dur_fr:.2f}s @ {out_sr_fr}Hz)\n"
        f"   ├─ Fonte PT-BR (Francisca Web): {dt_fr:6.1f} ms\n"
        f"   ├─ Trato Vocal RVC 40kHz GPU  : {m_fr['rvc_total_ms']:6.1f} ms\n"
        f"   └─ TOTAL FIM-A-FIM            : {tot_fr:6.1f} ms (RTF = {(tot_fr/1000.0)/dur_fr:.3f}x)"
    )


if __name__ == "__main__":
    asyncio.run(main())
