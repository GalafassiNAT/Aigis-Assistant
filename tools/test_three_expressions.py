"""
Test the 3 authentic Aigis vocal expressions in English:
1. Melancholic / Soft
2. Tactical / Combat
3. Conversational / Tartarus exploration (isolated from BGM via Demucs)
Using calibrated speed=0.95 and 32 ODE steps.
"""
import time
import os
import soundfile as sf
import numpy as np
import torch
from f5_tts.api import F5TTS

def test_expressions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Carregando F5-TTS no dispositivo: {device}...")
    f5 = F5TTS(device=device)
    
    test_phrase = "Good evening. All systems are fully operational and ready for your command."
    
    expressions = [
        {
            "name": "expression_melancholic",
            "audio": "models/voice_prompts/aigis_prompt_clean_mel1.wav",
            "txt": "models/voice_prompts/aigis_prompt_clean_mel1.txt"
        },
        {
            "name": "expression_tactical",
            "audio": "models/voice_prompts/aigis_prompt_tactical_clean.wav",
            "txt": "models/voice_prompts/aigis_prompt_tactical_clean.txt"
        },
        {
            "name": "expression_conversational",
            "audio": "models/voice_prompts/aigis_prompt_conversational.wav",
            "txt": "models/voice_prompts/aigis_prompt_conversational.txt"
        },
    ]
    
    os.makedirs("output/audio/tests_expressions", exist_ok=True)
    
    for exp in expressions:
        with open(exp["txt"], "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
            
        print(f"\n==========================================")
        print(f"[*] Testando Expressão: {exp['name']}")
        print(f"Ref Audio: {exp['audio']}")
        print(f"Ref Text:  \"{ref_text}\"")
        
        t0 = time.perf_counter()
        wav, sr, _ = f5.infer(
            ref_file=exp["audio"],
            ref_text=ref_text,
            gen_text=test_phrase,
            nfe_step=32,
            speed=0.95
        )
        latency = time.perf_counter() - t0
        audio_dur = len(wav) / sr
        
        peak = np.max(np.abs(wav))
        wav_norm = (wav / peak) * 0.89 if peak > 0 else wav
        
        out_path = f"output/audio/tests_expressions/{exp['name']}.wav"
        sf.write(out_path, wav_norm, sr)
        print(f"[+] Gerado em {latency:.2f}s | Duração: {audio_dur:.2f}s | Salvo: {out_path}")

if __name__ == "__main__":
    test_expressions()
