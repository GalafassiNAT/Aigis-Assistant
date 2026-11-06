"""
Test F5-TTS with clean prompt (6.4s) under 12s limit, with different speed factors (0.85, 0.95, 1.0)
and 32 vs 48 steps to eliminate rush and forced pacing.
"""
import time
import os
import soundfile as sf
import numpy as np
import torch
from f5_tts.api import F5TTS

def test_pacing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Carregando F5-TTS no dispositivo: {device}...")
    f5 = F5TTS(device=device)
    
    ref_audio = "models/voice_prompts/aigis_prompt_clean_mel1.wav"
    with open("models/voice_prompts/aigis_prompt_clean_mel1.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
        
    test_phrase = "Good evening. All systems are fully operational and ready for your command."
    
    os.makedirs("output/audio/tests_pacing", exist_ok=True)
    
    # Testar com speed 0.85 (mais calmo, pausado e natural) e 0.95
    configs = [
        {"speed": 0.85, "steps": 32, "name": "f5_clean_mel_speed085_steps32"},
        {"speed": 0.85, "steps": 48, "name": "f5_clean_mel_speed085_steps48"},
        {"speed": 0.95, "steps": 32, "name": "f5_clean_mel_speed095_steps32"},
    ]
    
    for cfg in configs:
        print(f"\n==========================================")
        print(f"[*] Testando speed={cfg['speed']} | steps={cfg['steps']}...")
        t0 = time.perf_counter()
        
        wav, sr, _ = f5.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=test_phrase,
            nfe_step=cfg["steps"],
            speed=cfg["speed"]
        )
        
        latency = time.perf_counter() - t0
        audio_dur = len(wav) / sr
        
        peak = np.max(np.abs(wav))
        wav_norm = (wav / peak) * 0.89 if peak > 0 else wav
        
        out_path = f"output/audio/tests_pacing/{cfg['name']}.wav"
        sf.write(out_path, wav_norm, sr)
        print(f"[+] Gerado em {latency:.2f}s | Duração: {audio_dur:.2f}s | Salvo: {out_path}")

if __name__ == "__main__":
    test_pacing()
