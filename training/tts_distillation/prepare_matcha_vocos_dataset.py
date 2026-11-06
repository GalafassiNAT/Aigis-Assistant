"""
Data preparation for Matcha-TTS + Vocos (24kHz, 100 mel channels).
Extracts Vocos mel-spectrogram features and Portuguese IPA phoneme tokens for all 250 sentences.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torchaudio
from phonemizer import phonemize
from vocos import Vocos
import matcha.text as mt
from matcha.utils.utils import intersperse

DATASET_DIR = Path("training/tts_distillation/dataset")
WAVS_DIR = DATASET_DIR / "wavs"
METADATA_CSV = DATASET_DIR / "metadata.csv"
OUTPUT_CACHE = DATASET_DIR / "matcha_vocos_dataset.pt"

def clean_ptbr_ipa(ipa_text: str) -> str:
    # espeak pt-br produces combining tildes for nasal vowels (e.g. ɐ̃)
    # Matcha's symbol vocabulary does not include U+0303
    return ipa_text.replace('\u0303', '')

def main():
    print("[*] Loading metadata...")
    df = pd.read_csv(METADATA_CSV, sep='|', header=None, names=['id', 'text'])
    print(f"Total items in metadata: {len(df)}")

    print("[*] Phonemizing Portuguese sentences to IPA via espeak...")
    texts = df['text'].tolist()
    raw_ipas = phonemize(
        texts,
        language='pt-br',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=True
    )

    cleaned_ipas = [clean_ptbr_ipa(ipa) for ipa in raw_ipas]

    # Verify all characters exist in matcha vocabulary
    for idx, (text, ipa) in enumerate(zip(texts, cleaned_ipas)):
        for c in ipa:
            if c not in mt._symbol_to_id:
                raise ValueError(f"Sentence {idx} has unknown char '{c}' (ord: {ord(c)}) in IPA: {ipa}")

    print("[*] Loading Vocos feature extractor (charactr/vocos-mel-24khz)...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    fe = vocos.feature_extractor

    data_entries = []
    all_mel_frames = []

    print("[*] Processing 250 audio files & extracting 100-mel features...")
    for idx, row in df.iterrows():
        sid = row['id']
        raw_text = row['text']
        ipa = cleaned_ipas[idx]

        wav_path = WAVS_DIR / f"{sid}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")

        wav, sr = torchaudio.load(str(wav_path))
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        
        # Vocos mel feature extraction: shape [1, 100, T]
        with torch.no_grad():
            mel = fe(wav).squeeze(0)  # shape [100, T]

        all_mel_frames.append(mel)

        # Convert IPA to token sequence and intersperse blanks (standard in Matcha-TTS)
        seq, _ = mt.text_to_sequence(ipa, [])
        seq_interspersed = intersperse(seq, 0)
        x_tensor = torch.IntTensor(seq_interspersed)

        data_entries.append({
            "id": sid,
            "text": raw_text,
            "ipa": ipa,
            "x": x_tensor,
            "mel": mel.cpu(),
        })

    # Compute dataset-level mel statistics
    cat_mels = torch.cat(all_mel_frames, dim=1)
    mel_mean = cat_mels.mean().item()
    mel_std = cat_mels.std().item()
    print(f"[+] Dataset mel statistics computed:")
    print(f"    mel_mean: {mel_mean:.6f}")
    print(f"    mel_std:  {mel_std:.6f}")

    # Train / Val Split:
    # Place aigis_0001 and aigis_0002 + random selection in validation set (15 items)
    # The rest (235 items) in training set
    val_ids = {"aigis_0001", "aigis_0002", "aigis_0010", "aigis_0025", "aigis_0050", 
               "aigis_0075", "aigis_0100", "aigis_0125", "aigis_0150", "aigis_0175",
               "aigis_0200", "aigis_0220", "aigis_0230", "aigis_0240", "aigis_0250"}

    train_data = []
    val_data = []

    for entry in data_entries:
        if entry["id"] in val_ids:
            val_data.append(entry)
        else:
            train_data.append(entry)

    print(f"[+] Split: {len(train_data)} train samples, {len(val_data)} validation samples.")

    save_payload = {
        "train": train_data,
        "val": val_data,
        "stats": {
            "mel_mean": mel_mean,
            "mel_std": mel_std,
            "n_feats": 100,
            "sample_rate": 24000,
            "n_fft": 1024,
            "hop_length": 256,
        }
    }

    torch.save(save_payload, OUTPUT_CACHE)
    file_size_mb = OUTPUT_CACHE.stat().st_size / (1024 * 1024)
    print(f"[+] Saved consolidated dataset to {OUTPUT_CACHE} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
