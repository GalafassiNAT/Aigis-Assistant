"""Renderizador Offline do Dataset do 'Teacher' (Aigis TTS + RVC).

Percorre o corpus_ptbr.txt, sintetiza cada frase com o pipeline atual da Aigis,
normaliza em 22050Hz Mono WAV 16-bit, remove silêncios mortos nas pontas (librosa)
e formata no padrão padrão LJSpeech para treinamento de Piper / VITS:
- dataset/wavs/0001.wav
- dataset/metadata.csv: 0001|Texto da frase
"""

import asyncio
import sys
import time
import zipfile
from pathlib import Path

# Adiciona a raiz do projeto no sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from core.voice import VoiceSynthesizer  # noqa: E402


def trim_silence(audio: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Remove silêncio inicial e final baseado em energia RMS."""
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    except ImportError:
        energy = np.abs(audio)
        threshold = np.max(energy) * (10 ** (-top_db / 20.0))
        indices = np.where(energy > threshold)[0]
        if len(indices) == 0:
            return audio
        return audio[indices[0]: indices[-1] + 1]


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Reamostra o áudio para 22050Hz (padrão ótimo do Piper VITS)."""
    if orig_sr == target_sr:
        return audio, target_sr
    try:
        import librosa
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled, target_sr
    except Exception:
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            resampled = signal.resample(audio, num_samples)
            return resampled.astype(np.float32), target_sr
        except Exception:
            return audio, orig_sr


async def render_dataset(corpus_path: str, output_dir: str):
    c_path = Path(corpus_path)
    if not c_path.exists():
        print(f"[ERRO] Corpus não encontrado em: {corpus_path}")
        return

    with open(c_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    out = Path(output_dir)
    wavs_dir = out / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = out / "metadata.csv"

    print("============================================================")
    print("🎙️ RENDERIZANDO DATASET AIGIS (LJSpeech Format)")
    print(f"Total de sentenças a sintetizar: {len(lines)}")
    print(f"Pasta de saída: {out.resolve()}")
    print("============================================================")

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

            # Sintetiza em array de memória com retry resiliente
            pcm = None
            sr = None
            t_s = time.time()
            for retry in range(3):
                try:
                    pcm, sr = await synth.synthesize_array(text)
                    break
                except Exception as e:
                    print(
                        f"\n⚠️ Falha temporária na sentença {idx} (tentativa {retry + 1}/3): {e}. Aguardando 2s..."
                    )
                    await asyncio.sleep(2.0)

            if pcm is None:
                print(f"❌ Não foi possível sintetizar a sentença {idx} após 3 tentativas. Pulando...")
                continue

            # Reamostragem para 22050Hz + corte de silêncio + normalização
            pcm, sr = resample_audio(pcm, sr, target_sr=22050)
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

        # Compacta em dataset.zip para envio ao Google Colab
        zip_path = out.parent / "dataset.zip"
        print(f"\n📦 Compactando dataset em {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for wav_file in wavs_dir.glob("*.wav"):
                zf.write(wav_file, arcname=f"dataset/wavs/{wav_file.name}")
            zf.write(metadata_csv, arcname="dataset/metadata.csv")

        total_time = time.time() - t0
        print(f"\n✅ Renderização concluída com sucesso em {total_time:.1f}s!")
        print(f"Arquivos gerados em: {wavs_dir}")
        print(f"Metadados em: {metadata_csv}")
        print(f"Arquivo zip pronto para o Colab: {zip_path}")

    finally:
        synth.close()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    c_file = script_dir / "corpus_ptbr.txt"
    dataset_folder = script_dir / "dataset"
    asyncio.run(render_dataset(str(c_file), str(dataset_folder)))
