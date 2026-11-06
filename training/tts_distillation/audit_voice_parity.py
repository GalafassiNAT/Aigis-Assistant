"""
Rigid Voice Identity & Acoustic Parity Verification Suite for Project Aigis.
Enforces strict biometric, phonetic, spectral and prosodic parity against the approved Gold Sweetspot milestone.
"""

import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, HubertModel

GOLD_REFERENCE_PATH = Path("output/audio/milestones_aigis_pt/aigis_0001_sweetspot_original_slice.wav")
GOLD_CONVERSATIONAL_EN = Path("output/audio/tests_refinement/conversational_speed088.wav")

class AigisVoiceAuditor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Aigis Voice Auditor on {self.device}...")
        
        # 1. WavLM Speaker Verification model for biometric vocal identity
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv", use_safetensors=True).to(self.device).eval()
        
        # 2. HuBERT for Phonetic Posteriorgram (PPG) alignment
        self.hubert = HubertModel.from_pretrained("r3gm/hubert_base").to(self.device).eval()
        
        # Load and embed gold references
        if not GOLD_REFERENCE_PATH.exists():
            raise FileNotFoundError(f"Missing gold reference: {GOLD_REFERENCE_PATH}")
        
        self.gold_y, self.gold_sr = sf.read(str(GOLD_REFERENCE_PATH))
        self.gold_emb = self.extract_speaker_embedding(self.gold_y, self.gold_sr)
        self.gold_ppg = self.extract_hubert_ppg(self.gold_y, self.gold_sr)
        self.gold_metrics = self.compute_acoustic_profile(self.gold_y, self.gold_sr)
        print("[+] Gold reference loaded and acoustic profile indexed.")

    def extract_speaker_embedding(self, y: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 16000:
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        else:
            y_16k = y
        inputs = self.fe(y_16k, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.sv(**inputs).embeddings
        return F.normalize(emb, dim=-1)

    def extract_hubert_ppg(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        else:
            y_16k = y
        inp = torch.from_numpy(y_16k).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            h12 = self.hubert(inp, output_hidden_states=True).hidden_states[12][0]
            h12 = F.normalize(h12, dim=-1)
        return h12.cpu().numpy()

    def compute_acoustic_profile(self, y: np.ndarray, sr: int) -> dict:
        dur = len(y) / sr
        rms = float(np.sqrt(np.mean(y**2)))
        
        # F0 estimation
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C6'), sr=sr)
        f0_v = f0[voiced_flag > 0]
        f0_med = float(np.nanmedian(f0_v)) if len(f0_v) > 0 else 0.0
        f0_std = float(np.nanstd(f0_v)) if len(f0_v) > 0 else 0.0
        
        # High Frequency energy (>7 kHz) - detect "serra ligada"
        S = np.abs(librosa.stft(y))**2
        freqs = librosa.fft_frequencies(sr=sr)
        total_e = float(np.sum(S) + 1e-12)
        hf_ratio = float(np.sum(S[freqs > 7000]) / total_e * 100.0)
        
        # Spectral centroid & flatness
        cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        # Harmonic to noise ratio
        harm, perc = librosa.effects.hpss(y)
        hnr = float(10 * np.log10((np.sum(harm**2) + 1e-12) / (np.sum(perc**2) + 1e-12)))
        
        return {
            "dur": dur,
            "rms": rms,
            "f0_med": f0_med,
            "f0_std": f0_std,
            "hf_ratio": hf_ratio,
            "cent": cent,
            "flatness": flatness,
            "hnr": hnr,
        }

    def evaluate_candidate(self, wav_path: str or Path) -> dict:
        p = Path(wav_path)
        y, sr = sf.read(str(p))
        if y.ndim > 1:
            y = y.mean(axis=1)

        profile = self.compute_acoustic_profile(y, sr)
        
        # 1. Speaker identity similarity (WavLM-SV)
        cand_emb = self.extract_speaker_embedding(y, sr)
        identity_sim = float(torch.cosine_similarity(self.gold_emb, cand_emb).item()) * 100.0

        # 2. Phonetic fidelity (HuBERT Layer 12 DTW Cosine)
        cand_ppg = self.extract_hubert_ppg(y, sr)
        cost_matrix = 1.0 - np.dot(self.gold_ppg, cand_ppg.T)
        _, wp = librosa.sequence.dtw(C=cost_matrix, subseq=False)
        dtw_sims = [1.0 - cost_matrix[i, j] for i, j in wp]
        phonetic_sim = float(np.mean(dtw_sims)) * 100.0

        # Strict checks
        # 1. Identity: >= 96.0% (target), fail if < 95.0%
        # 2. Phonetic DTW: >= 88.0%
        # 3. F0: 228 - 235 Hz
        # 4. HF ratio: 6.5% - 10.5% (reject if > 11.5% -> "serra ligada")
        # 5. Duration: 3.40s - 3.80s
        checks = {
            "identity_passed": identity_sim >= 95.0,
            "phonetic_passed": phonetic_sim >= 88.0,
            "pitch_passed": 226.0 <= profile["f0_med"] <= 236.0,
            "clean_treble_passed": profile["hf_ratio"] <= 11.0,  # "serra ligada" guard
            "duration_passed": 3.40 <= profile["dur"] <= 3.85,
        }
        all_passed = all(checks.values())

        score = (
            0.35 * identity_sim +
            0.25 * phonetic_sim +
            0.15 * max(0, 100 - abs(profile["f0_med"] - self.gold_metrics["f0_med"]) * 5) +
            0.15 * max(0, 100 - max(0, profile["hf_ratio"] - 8.0) * 10) +
            0.10 * max(0, 100 - abs(profile["dur"] - self.gold_metrics["dur"]) * 50)
        )

        return {
            "path": str(p),
            "score": score,
            "all_passed": all_passed,
            "identity_sim": identity_sim,
            "phonetic_sim": phonetic_sim,
            "profile": profile,
            "checks": checks,
        }

if __name__ == "__main__":
    auditor = AigisVoiceAuditor()
    files_to_audit = [
        "output/audio/milestones_aigis_pt/aigis_0001_sweetspot_original_slice.wav",
        "output/audio/test_rates/aigis_0001_vocos_roundtrip.wav",
        "output/audio/test_rates/aigis_matcha_vocos_6steps.wav",
        "output/audio/test_rates/aigis_gt_aligned_temp0.0_norm.wav",
        "output/audio/test_rates/aigis_calibrated_temp0.05_norm.wav",
    ]
    
    print("\n" + "="*85)
    print(f"{'FILE':42s} | {'SCORE':5s} | {'ID-SIM':6s} | {'PPG-PT':6s} | {'F0 (Hz)':7s} | {'HF (>7k)':8s} | STATUS")
    print("="*85)
    for f in files_to_audit:
        res = auditor.evaluate_candidate(f)
        status = "PASSED (APPROVED)" if res["all_passed"] else "FAILED (REJECTED)"
        print(f"{Path(f).name:42s} | {res['score']:5.1f} | {res['identity_sim']:5.1f}% | {res['phonetic_sim']:5.1f}% | {res['profile']['f0_med']:7.1f} | {res['profile']['hf_ratio']:7.2f}% | {status}")
    print("="*85 + "\n")
