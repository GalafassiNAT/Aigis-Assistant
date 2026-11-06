"""
Test F5-TTS with native English phrases and audio normalization to check authentic acoustic quality and eliminate clipping.
"""
import time
import os
import soundfile as sf
import numpy as np
import torch
from f5_tts.api import F5TTS

def test_english():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Carregando F5-TTS no dispositivo: {device}...")
    f5 = F5TTS(device=device)
    
    ref_audio = "models/voice_prompts/aigis_prompt_tactical.wav"
    with open("models/voice_prompts/aigis_prompt_tactical.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
        
    test_phrases = [
        "Good evening. All systems are fully operational and ready for your command.",
        "Understood. I will protect you with all my strength. Initiating combat mode.",
        "The fact that a future exists gives us reason to keep moving forward."
    ]
    
    os.makedirs("output/audio/tests", exist_ok=True)
    
    for i, gen_text in enumerate(test_phrases):
        print(f"\n--- Teste Inglês {i+1} ---")
        print(f"Texto alvo: \"{gen_text}\"")
        t0 = time.perf_counter()
        
        wav, sr, _ = f5.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            nfe_step=32,
            speed=1.0
        )
        
        latency = time.perf_counter() - t0
        audio_dur = len(wav) / sr
        rtf = latency / audio_dur if audio_dur > 0 else 0
        
        # Prevenir estouro / clipping normalizando para -1.0 dBFS (0.89 max amplitude)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav_normalized = (wav / peak) * 0.89
        else:
            wav_normalized = wav
            
        out_path = f"output/audio/tests/f5_test_en_{i+1}.wav"
        sf.write(out_path, wav_normalized, sr)
        print(f"[+] Gerado em {latency:.2f}s | Duração: {audio_dur:.2f}s | Salvo em: {out_path} (Pico normalizado: {np.max(np.abs(wav_normalized)):.2f})")

if __name__ == "__main__":
    test_english()
