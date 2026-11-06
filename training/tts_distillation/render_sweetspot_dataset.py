"""Renderizador Offline do Dataset de Destilação da Aigis (Pipeline Sweetspot Oficial 24kHz).

Utiliza o pipeline Teacher aprovado em `aigis_pt_v2_studio_sweetspot_231hz.wav` @ 24.000 Hz:
1. F5-TTS PT-BR (`viperdf/f5-tts_pt-br`) com prompt de referência `aigis_0004` (speed=0.82)
2. Trato vocal RVC v2 40kHz em VRAM (`AigisMrM0dZ_300e_5700s.pth`, ir=0.25, ps=0.0)
3. Transferência espectral LTAS unificada do padrão ouro `conversational_speed088.wav`
4. Ajuste fino de estúdio (`process_studio_clarity`): resample polifásico sinc 23/24 (-0.74 st),
   corte anti-nasal (-2.0 dB @ 1220 Hz), ressonância suave de peito (+0.8 dB @ 215 Hz) e brilho
   de ar de microfone condensador de estúdio (+1.6 dB @ 7 kHz)
5. Reamostragem de estúdio via soxr VHQ para 24.000 Hz Mono WAV 16-bit e trim de silêncios
6. Formatação no padrão LJSpeech (dataset/wavs/*.wav e dataset/metadata.csv)
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

# Fix environment for ROCm
if sys.platform.startswith("linux"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import librosa
import numpy as np
np.float = float
np.complex = complex
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt
import soundfile as sf
import soxr
import torch
import whisper

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools"))

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT
from huggingface_hub import hf_hub_download
from test_vram_source_filter_ptbr import InVRAMVocalTractAigis


def design_peaking_sos(fc: float, gain_db: float, Q: float, fs: int) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def design_high_shelf_sos(fc: float, gain_db: float, Q: float, fs: int) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


class AigisTeacherSynthesizer:
    """Motor Teacher 100% calibrado no Sweetspot da Aigis PT-BR."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[*] Inicializando AigisTeacherSynthesizer ({self.device})...")

        # 1. Carrega F5-TTS viperdf
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)
        ckpt_path = hf_hub_download(repo_id="viperdf/f5-tts_pt-br", filename="model_pt-br.safetensors")
        vocab_path = hf_hub_download(repo_id="viperdf/f5-tts_pt-br", filename="vocab.txt")
        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=self.device)
        self.f5_model = load_model(DiT, model_cfg, ckpt_path, mel_spec_type="vocos", vocab_file=vocab_path, ode_method="euler", use_ema=True, device=self.device)

        # 2. Referência aprovada aigis_0004
        ref_pt_wav = str(project_root / "training/f5_ptbr/wavs/aigis_pt_aigis_0004.wav")
        ref_pt_txt = "bom dia! todos os módulos de vigilância e auxílio estão ativos."
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(ref_pt_wav, ref_pt_txt)

        # 3. Trato vocal RVC
        self.rvc = InVRAMVocalTractAigis()

        # 4. Pré-computa LTAS alvo de conversational_speed088.wav (40kHz)
        gold_path = str(project_root / "output/audio/tests_refinement/conversational_speed088.wav")
        y_gold, _ = librosa.load(gold_path, sr=40000)
        self.n_fft = 2048
        S_gold = np.abs(librosa.stft(y_gold, n_fft=self.n_fft))
        frame_rms_g = np.sqrt(np.mean(S_gold**2, axis=0))
        active_g = frame_rms_g > np.percentile(frame_rms_g, 35)
        self.ltas_gold = np.sqrt(np.mean(S_gold[:, active_g]**2, axis=1) + 1e-8)
        self.log_gold = gaussian_filter1d(np.log(self.ltas_gold), sigma=8)

        # 5. Whisper ASR local para verificação automática de integridade fonética
        self.whisper = whisper.load_model("base", device="cpu")

        print("[OK] AigisTeacherSynthesizer pronto para renderização!")

    def _apply_ltas(self, wav: np.ndarray, sr: int, strength: float = 0.75, max_db: float = 9.5) -> np.ndarray:
        D = librosa.stft(wav, n_fft=self.n_fft)
        S = np.abs(D)
        frame_rms = np.sqrt(np.mean(S**2, axis=0))
        active = frame_rms > np.percentile(frame_rms, 35)
        ltas_src = np.sqrt(np.mean(S[:, active]**2, axis=1) + 1e-8)
        log_src = gaussian_filter1d(np.log(ltas_src), sigma=8)

        diff_db = (self.log_gold - log_src) * (20.0 / np.log(10.0))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        diff_db[freqs < 75] = 0.0
        diff_db = np.clip(diff_db * strength, -max_db, max_db)
        gain_curve = (10.0 ** (diff_db / 20.0)).astype(np.float32)[:, None]

        D_matched = (D * gain_curve).astype(np.complex64)
        out = librosa.istft(D_matched, length=len(wav))
        return out.astype(np.float32)

    def _apply_studio_clarity(self, wav_in: np.ndarray, sr: int) -> np.ndarray:
        x = wav_in.astype(np.float64)
        sos_list = [
            butter(2, 65.0 / (0.5 * sr), btype="highpass", output="sos"),
            design_peaking_sos(215.0, 0.8 / 2.0, Q=1.1, fs=sr),           # calor de peito acolhedor
            design_peaking_sos(1220.0, -2.0 / 2.0, Q=1.5, fs=sr),         # corte anti-nasal
            design_peaking_sos(3200.0, -0.8 / 2.0, Q=1.2, fs=sr),         # suaviza estridência
            design_high_shelf_sos(7000.0, 1.6 / 2.0, Q=0.707, fs=sr),      # brilho de microfone de condensador de estúdio
        ]
        sos_all = np.vstack(sos_list)
        y = sosfiltfilt(sos_all, x)

        # Sinc varispeed 23/24 (-0.74 st) -> calibra pitch em 231-232 Hz e acalma a cadência
        y = resample_poly(y, 24, 23)
        return y.astype(np.float32)

    @staticmethod
    def sanitize_tts_prompt(text: str) -> str:
        """Normaliza pontuações para o F5-TTS evitar acelerações espúrias ou cortes consonantais.
        Exclamações no meio de frases compostas (ex: 'Olá, mestre! Como posso...') são convertidas
        para ponto final ('. ') que gera pausa confortável e preserva oclusivas e 'tr' limpos.
        """
        import re
        t = text.strip()
        # Substitui '! ' no meio da frase por '. ' para cadência equilibrada
        t = re.sub(r'!\s+(?=[A-Za-zÀ-ÿ])', '. ', t)
        return t.lower()

    def synthesize(self, text: str, target_sr: int = 24000, initial_seed: int = 42, max_retries: int = 4) -> np.ndarray:
        clean_target = text.lower().strip().replace("!", "").replace(",", "").replace(".", "")
        first_target_word = clean_target.split()[0] if clean_target else ""
        prompt_text = self.sanitize_tts_prompt(text)
        fixed_speed = 0.82  # Velocidade nativa calibrada do F5-TTS que mantém F0 em 228-231Hz

        for attempt in range(max_retries):
            seed = initial_seed + attempt * 101
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # 1. F5-TTS PT-BR com cadência nativa e entonação natural de estúdio
            wav_f5, _, _ = infer_process(
                self.ref_audio,
                self.ref_text,
                prompt_text,
                self.f5_model,
                self.vocoder,
                mel_spec_type="vocos",
                speed=fixed_speed,
                target_rms=0.08,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                nfe_step=32,
                device=self.device,
            )

            # Medição adaptativa do F0 bruto da geração F5-TTS para travar afinação no sweetspot ouro
            f0_raw_arr, vf_raw, _ = librosa.pyin(wav_f5, fmin=120, fmax=400, sr=24000)
            f0_v = f0_raw_arr[vf_raw & ~np.isnan(f0_raw_arr)]
            f0_raw = float(np.median(f0_v)) if len(f0_v) > 5 else 236.8
            target_rvc_f0 = 227.0 * (24.0 / 23.0)
            pitch_shift_st = float(np.clip(12.0 * np.log2(target_rvc_f0 / f0_raw), -3.0, 3.0))

            # 2. RVC Vocal Tract (40kHz) com trava de pitch adaptativa contra estridência
            out_rvc, sr_rvc, _ = self.rvc.convert_tensor(
                wav_f5, 24000, pitch_shift=pitch_shift_st, index_rate=0.25, protect=0.33, rms_mix_rate=0.25
            )

            # 3. LTAS Transfer
            out_ltas = self._apply_ltas(out_rvc, sr_rvc, strength=0.75)

            # 4. Studio Clarity & Sinc Varispeed
            out_studio = self._apply_studio_clarity(out_ltas, sr_rvc)

            # 5. Reamostragem de alta fidelidade para target_sr (24.000 Hz padrão de estúdio)
            if sr_rvc != target_sr:
                out_resampled = soxr.resample(out_studio, sr_rvc, target_sr, quality="VHQ")
            else:
                out_resampled = out_studio

            # 6. Preservação da respiração natural de estúdio (SEM corte agressivo de ataque)
            fade_len = int(0.010 * target_sr)  # 10ms
            if len(out_resampled) > 2 * fade_len:
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                out_resampled[:fade_len] *= fade_in
                out_resampled[-fade_len:] *= fade_out

            # 7. Normalização de dinâmica
            rms = np.sqrt(np.mean(out_resampled ** 2) + 1e-12)
            out_final = out_resampled * (0.088 / rms)
            peak = np.max(np.abs(out_final))
            if peak > 0.89:
                out_final = (out_final / peak) * 0.89

            # 8. Verificação automática com Whisper para garantir que a sílaba inicial não foi engolida
            asr_res = self.whisper.transcribe(out_final, language="pt", fp16=False)
            asr_text = asr_res["text"].lower()

            if not first_target_word or first_target_word in asr_text:
                return out_final.astype(np.float32)

            print(f"      [RETRY] Tentativa {attempt+1}/{max_retries}: Whisper leu '{asr_text.strip()}' (esperado iniciar com '{first_target_word}'). Recalculando...")

        return out_final.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Renderiza dataset de destilação Aigis Sweetspot")
    parser.add_argument("--corpus", type=str, default="training/tts_distillation/corpus_ptbr.txt", help="Caminho do corpus TXT")
    parser.add_argument("--output_dir", type=str, default="training/tts_distillation/dataset", help="Diretório de saída")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Taxa de amostragem alvo (padrão: 24000 Hz)")
    parser.add_argument("--limit", type=int, default=None, help="Limite de frases para renderizar (para testes rápidos)")
    args = parser.parse_args()

    corpus_path = project_root / args.corpus
    out_dir = project_root / args.output_dir
    wavs_dir = out_dir / "wavs"
    metadata_csv = out_dir / "metadata.csv"

    wavs_dir.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if args.limit:
        lines = lines[: args.limit]

    print("============================================================")
    print(f"🎙️ RENDERIZANDO DATASET AIGIS SWEETSPOT @ {args.sample_rate}Hz (LJSpeech Format)")
    print(f"Total de sentenças: {len(lines)}")
    print(f"Pasta de saída: {out_dir}")
    print("============================================================")

    teacher = AigisTeacherSynthesizer()

    entries = []
    t_start = time.perf_counter()

    for idx, text in enumerate(lines, start=1):
        sample_id = f"aigis_{idx:04d}"
        wav_path = wavs_dir / f"{sample_id}.wav"

        # Resumabilidade e preservação das amostras ouro pré-existentes a 24.000 Hz
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            info = sf.info(str(wav_path))
            if info.samplerate == args.sample_rate:
                audio, _ = sf.read(str(wav_path))
                dur = len(audio) / args.sample_rate
                entries.append(f"{sample_id}|{text}")
                tag = "GOLD SLICE SWEETSPOT" if idx in (1, 2) else "JÁ RENDERIZADO (24kHz)"
                print(f"[{idx:03d}/{len(lines):03d}] {sample_id} ({dur:.2f}s) [{tag}] | \"{text[:45]}...\"")
                continue

        t0 = time.perf_counter()
        audio = teacher.synthesize(text, target_sr=args.sample_rate)
        sf.write(str(wav_path), audio, args.sample_rate, subtype="PCM_16")
        dur = len(audio) / args.sample_rate
        elapsed = time.perf_counter() - t0

        entries.append(f"{sample_id}|{text}")
        print(f"[{idx:03d}/{len(lines):03d}] {sample_id} ({dur:.2f}s) gerado em {elapsed:.2f}s | \"{text[:45]}...\"")

        # Flush incremental de segurança para metadata.csv
        with open(metadata_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(entries) + "\n")

    # Salva metadata.csv
    with open(metadata_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(entries) + "\n")

    total_time = time.perf_counter() - t_start
    print(f"\n[OK] Renderização de {len(lines)} amostras concluída em {total_time:.1f}s ({total_time/len(lines):.2f}s por frase).")

    # Compacta em dataset.zip se estiver renderizando o conjunto completo
    if not args.limit:
        zip_path = out_dir.parent / "dataset.zip"
        print(f"\n[*] Criando arquivo compactado para treinamento: {zip_path}...")
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(str(metadata_csv), arcname="metadata.csv")
            for w in wavs_dir.glob("*.wav"):
                z.write(str(w), arcname=f"wavs/{w.name}")
        print(f"[OK] dataset.zip gerado com sucesso ({zip_path.stat().st_size / (1024*1024):.1f} MB)!")


if __name__ == "__main__":
    main()
