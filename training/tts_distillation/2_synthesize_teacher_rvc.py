"""Renderizador Offline do Dataset do 'Teacher' (Aigis TTS + RVC).

Percorre o corpus_ptbr.txt, sintetiza cada frase com o pipeline atual da Aigis,
normaliza em 22050Hz Mono WAV 16-bit, remove silêncios mortos nas pontas (librosa)
e formata no padrão padrão LJSpeech para treinamento de Piper / VITS:
- dataset/wavs/0001.wav
- dataset/metadata.csv: 0001|Texto da frase
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Adiciona a raiz do projeto no sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import soundfile as sf
from core.voice import VoiceSynthesizer


def trim_silence(audio: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Remove silêncio inicial e final baseado em energia RMS."""
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    except ImportError:
        # Fallback simples caso librosa não esteja instalado
        energy = np.abs(audio)
        threshold = np.max(energy) * (10 ** (-top_db / 20.0))
        indices = np.where(energy > threshold)[0]
        if len(indices) == 0:
            return audio
        return audio[indices[0]: indices[-1] + 1]


async def render_dataset(corpus_path: str, output_dir: str):
    c_path = Path(corpus_path)
    if not c_path.exists():
        print(f"[ERRO] Corpus não encontrado em: {corpus_path}")
        return

    with open(c_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    out = Path(output_dir)
    wavs_dir = out / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = out / "metadata.csv"

    print(f"============================================================")
    print(f"🎙️ RENDERIZANDO DATASET AIGIS (LJSpeech Format)")
    print(f"Total de sentenças a sintetizar: {len(lines)}")
    print(f"Pasta de saída: {out.resolve()}")
    print(f"============================================================")

    synth = VoiceSynthesizer()
    entries = []

    try:
        t0 = time.time()
        for idx, text in enumerate(lines, start=1):
            sample_id = f"aigis_{idx:04d}"
            target_wav = wavs_dir / f"{sample_id}.wav"

            if target_wav.exists():
                entries.append(f"{sample_id}|{text}")
                continue

            # Sintetiza em array de memória sem bloquear
            t_s = time.time()
            pcm, sr = await synth.synthesize_array(text)

            # Normalização e corte de silêncio
            pcm = trim_silence(pcm, top_db=32.0)
            max_val = np.max(np.abs(pcm))
            if max_val > 0:
                pcm = pcm / max_val * 0.95

            # Salva em WAV 16-bit PCM
            sf.write(str(target_wav), pcm, sr, format="WAV", subtype="PCM_16")
            entries.append(f"{sample_id}|{text}")

            dt = (time.time() - t_s) * 1000
            print(f"[{idx}/{len(lines)}] '{sample_id}.wav' sintetizado ({dt:.0f}ms): {text[:45]}...")

        with open(metadata_csv, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(entry + "\n")

        total_time = time.time() - t0
        print(f"\n✅ Renderização concluída com sucesso em {total_time:.1f}s!")
        print(f"Arquivos gerados em: {wavs_dir}")
        print(f"Metadados em: {metadata_csv}")

    finally:
        synth.close()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    c_file = script_dir / "corpus_ptbr.txt"
    dataset_folder = script_dir / "dataset"
    asyncio.run(render_dataset(str(c_file), str(dataset_folder)))
