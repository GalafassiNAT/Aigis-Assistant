"""Compilador de Dataset para Fine-Tuning do F5-TTS (Aigis PT-BR).

Combina:
1. CML-TTS Português (726 amostras de estúdio em 24kHz entre 2s e 14s) para maestria fonética do PT-BR.
2. Segmentos autênticos da Dawn Bennett (Persona 3 Reload) para calibração harmônica e timbre original.
3. Frases assistente da Aigis (250 frases do corpus PT-BR).

Saídas:
- training/f5_ptbr/wavs/*.wav (24.000 Hz Mono, normalizado RMS=0.08)
- training/f5_ptbr/metadata.csv (audio_file|text)
- Executa a pré-compilação em Arrow para treinamento instantâneo via prepare_csv_wavs.
"""

import io
import os
import re
import sys
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm

TARGET_SR = 24000
TARGET_RMS = 0.08

OUTPUT_DIR = Path("training/f5_ptbr")
WAVS_DIR = OUTPUT_DIR / "wavs"
PREPPED_DIR = OUTPUT_DIR / "prepped"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"

PARQUET_PATH = Path(
    "/home/nora/.cache/huggingface/hub/datasets--freds0--cml_tts_dataset_portuguese/snapshots/f016b76dd2e2c0aaf5f96ff78bd65601f8c4068b/data/dev-00000-of-00001.parquet"
)
AIGIS_RAW_MP3 = Path("training/voice_sources/raw/Aigis English Voice Lines.mp3")
ASSISTANT_DIR = Path("training/tts_distillation/dataset")


def clean_text(text: str) -> str:
    """Limpa e formata o texto para manter apenas caracteres permitidos."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_audio(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """Normaliza o áudio para um RMS alvo sem distorção e sem clipping digital."""
    rms = np.sqrt(np.mean(audio**2) + 1e-9)
    if rms > 0:
        audio = audio * (target_rms / rms)
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32)


def process_cml_portuguese(max_samples: int = 700):
    """Extrai amostras de estúdio em português com 24kHz nativos."""
    print(f"\n[1/3] Processando CML-TTS Português ({PARQUET_PATH.name})...")
    if not PARQUET_PATH.exists():
        print(f"Erro: Parquet não encontrado em {PARQUET_PATH}")
        return []

    table = pq.read_table(PARQUET_PATH)
    rows = table.to_pylist()
    valid_pairs = []

    count = 0
    for idx, row in enumerate(tqdm(rows, desc="Extraindo CML-TTS")):
        dur = row["duration"]
        if dur < 2.5 or dur > 13.0:
            continue
        text = clean_text(row["transcript"])
        if len(text) < 10:
            continue

        audio_bytes = row["audio"]["bytes"]
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        # Reamostragem se necessário
        if sr != TARGET_SR:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=-1)

        # Trim silêncios
        audio_data, _ = librosa.effects.trim(audio_data, top_db=35)
        audio_data = normalize_audio(audio_data)

        out_name = f"cml_pt_{count:04d}.wav"
        out_path = WAVS_DIR / out_name
        sf.write(out_path, audio_data, TARGET_SR)

        valid_pairs.append((str(out_path.resolve()), text))
        count += 1
        if count >= max_samples:
            break

    print(f"-> CML-TTS: {len(valid_pairs)} amostras salvas.")
    return valid_pairs


def process_aigis_english(whisper_model=None):
    """Segmenta o áudio autêntico da Dawn Bennett com timestamps e transcrição."""
    print(f"\n[2/3] Processando falas autênticas da Dawn Bennett ({AIGIS_RAW_MP3.name})...")
    if not AIGIS_RAW_MP3.exists():
        print("Aviso: Áudio da Dawn Bennett não encontrado.")
        return []

    import whisper

    if whisper_model is None:
        whisper_model = whisper.load_model("base", device="cpu")

    res = whisper_model.transcribe(str(AIGIS_RAW_MP3), language="en")
    raw_audio, sr = librosa.load(str(AIGIS_RAW_MP3), sr=TARGET_SR, mono=True)

    valid_pairs = []
    count = 0
    for seg in tqdm(res["segments"], desc="Segmentando Dawn Bennett"):
        dur = seg["end"] - seg["start"]
        text = clean_text(seg["text"])
        if dur < 1.8 or dur > 10.0 or len(text) < 8:
            continue

        start_sample = int(seg["start"] * TARGET_SR)
        end_sample = int(seg["end"] * TARGET_SR)
        audio_chunk = raw_audio[start_sample:end_sample]

        audio_chunk, _ = librosa.effects.trim(audio_chunk, top_db=30)
        audio_chunk = normalize_audio(audio_chunk)

        out_name = f"dawn_bennett_{count:04d}.wav"
        out_path = WAVS_DIR / out_name
        sf.write(out_path, audio_chunk, TARGET_SR)

        valid_pairs.append((str(out_path.resolve()), text))
        count += 1

    print(f"-> Dawn Bennett (Autêntico): {len(valid_pairs)} amostras salvas.")
    return valid_pairs


def process_assistant_ptbr():
    """Importa as 250 frases de assistente específicas da Aigis em PT-BR."""
    print("\n[3/3] Processando frases da Persona Aigis em PT-BR...")
    meta_path = ASSISTANT_DIR / "metadata.csv"
    if not meta_path.exists():
        print("Aviso: Metadata de assistente não encontrado.")
        return []

    valid_pairs = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 1)
            audio_id, text = parts[0], clean_text(parts[1])
            wav_file = ASSISTANT_DIR / "wavs" / f"{audio_id}.wav"
            if not wav_file.exists():
                continue

            audio_data, sr = sf.read(str(wav_file))
            if sr != TARGET_SR:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=-1)

            audio_data, _ = librosa.effects.trim(audio_data, top_db=32)
            audio_data = normalize_audio(audio_data)

            out_name = f"aigis_pt_{audio_id}.wav"
            out_path = WAVS_DIR / out_name
            sf.write(out_path, audio_data, TARGET_SR)

            valid_pairs.append((str(out_path.resolve()), text))

    print(f"-> Aigis Persona PT-BR: {len(valid_pairs)} amostras salvas.")
    return valid_pairs


def main():
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    PREPPED_DIR.mkdir(parents=True, exist_ok=True)

    if METADATA_CSV.exists() and len(list(WAVS_DIR.glob("*.wav"))) >= 900:
        print(f"-> Arquivos de áudio já extraídos ({len(list(WAVS_DIR.glob('*.wav')))} WAVs). Reutilizando {METADATA_CSV}...")
    else:
        pairs = []
        pairs.extend(process_cml_portuguese(max_samples=700))
        pairs.extend(process_aigis_english())
        pairs.extend(process_assistant_ptbr())

        print(f"\n==================================================")
        print(f"Total de pares reunidos para o Fine-Tuning: {len(pairs)}")
        print(f"Gravando {METADATA_CSV} ...")

        with open(METADATA_CSV, "w", encoding="utf-8") as f:
            f.write("audio_file|text\n")
            for audio_file, text in pairs:
                f.write(f"{audio_file}|{text}\n")

        print(f"Metadata gravado com sucesso! ({len(pairs)} registros)")

    # Pré-compilação com f5_tts.train.datasets.prepare_csv_wavs
    print("\nExecutando pré-compilação Arrow para F5-TTS...")
    import f5_tts.train.datasets.prepare_csv_wavs as prep_module
    from importlib.resources import files

    actual_vocab_path = Path(str(files("f5_tts").joinpath("infer/examples/vocab.txt")))
    prep_module.PRETRAINED_VOCAB_PATH = actual_vocab_path

    prep_module.prepare_and_save_set(
        inp_dir=str(METADATA_CSV.resolve()),
        out_dir=str(PREPPED_DIR.resolve()),
        is_finetune=True,
        num_workers=4,
    )
    print("\n✅ Dataset do F5-TTS totalmente preparado e indexado em Arrow!")


if __name__ == "__main__":
    main()
