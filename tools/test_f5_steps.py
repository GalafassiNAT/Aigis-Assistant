"""
Test F5-TTS with higher ODE steps (48, 64, 80) and test Melancholic vs Tactical prompts
to evaluate max potential quality before distillation.
"""
import time
import os
import soundfile as sf
import numpy as np
import torch
from f5_tts.api import F5TTS

def test_steps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Carregando F5-TTS no dispositivo: {device}...")
    f5 = F5TTS(device=device)
    
    ref_audio = "models/voice_prompts/aigis_prompt_tactical.wav"
    with open("models/voice_prompts/aigis_prompt_tactical.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
        
    test_phrase = "Good evening. All systems are fully operational and ready for your command."
    
    os.makedirs("output/audio/tests_steps", exist_ok=True)
    
    # Testar 48 e 64 passos
    for steps in [48, 64]:
        print(f"\n==========================================")
        print(f"[*] Testando {steps} passos de ODE (Euler/Midpoint)...")
        t0 = time.perf_counter()
        
        wav, sr, _ = f5.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=test_phrase,
            nfe_step=steps,
            speed=1.0
        )
        
        latency = time.perf_counter() - t0
        audio_dur = len(wav) / sr
        
        # Normalização -1.0 dBFS para evitar saturação
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav_norm = (wav / peak) * 0.89
        else:
            wav_norm = wav
            
        out_path = f"output/audio/tests_steps/f5_steps_{steps}.wav"
        sf.write(out_path, wav_norm, sr)
        print(f"[+] Gerado com {steps} passos em {latency:.2f}s | Duração: {audio_dur:.2f}s | Salvo: {out_path}")

    # Também testar com o prompt melancólico (tom mais natural e calmo) com 64 passos
    print(f"\n==========================================")
    print(f"[*] Testando com Prompt Melancólico (64 passos)...")
    ref_mel = "models/voice_prompts/aigis_prompt_melancholic.wav"
    with open("models/voice_prompts/aigis_prompt_melancholic.txt", "r", encoding="utf-8") as f:
        ref_mel_text = f.read().strip()
        
    t0 = time.perf_counter()
    wav_mel, sr, _ = f5.infer(
        ref_file=ref_mel,
        ref_text=ref_mel_text,
        gen_text=test_phrase,
        nfe_step=64,
        speed=1.0
    )
    lat_mel = time.perf_counter() - t0
    peak = np.max(np.abs(wav_mel))
    wav_mel_norm = (wav_mel / peak) * 0.89 if peak > 0 else wav_mel
    out_mel = "output/audio/tests_steps/f5_prompt_melancholic_64.wav"
    sf.write(out_mel, wav_mel_norm, sr)
    print(f"[+] Gerado Prompt Melancólico em {lat_mel:.2f}s | Salvo: {out_mel}")

if __name__ == "__main__":
    test_steps()
