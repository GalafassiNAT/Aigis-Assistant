"""
Test F5-TTS Zero-Shot Cross-Lingual Voice Cloning for Aigis.
Clones Dawn Bennett's English vocal timbre/formants directly to Brazilian Portuguese text.
"""
import time
import os
import soundfile as sf
import torch
from f5_tts.api import F5TTS

def test_cloning():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inicializando F5-TTS no dispositivo: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})...")
    
    start_load = time.perf_counter()
    f5 = F5TTS(device=device)
    print(f"[+] Modelo F5-TTS carregado em {time.perf_counter() - start_load:.2f}s")
    
    ref_audio = "models/voice_prompts/aigis_prompt_tactical.wav"
    with open("models/voice_prompts/aigis_prompt_tactical.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
        
    test_phrases = [
        "Boa noite, mestre! Sistemas operacionais prontos e totalmente calibrados.",
        "Entendido. Aguardo novas instruções para o combate."
    ]
    
    os.makedirs("output/audio/tests", exist_ok=True)
    
    for i, gen_text in enumerate(test_phrases):
        print(f"\n--- Teste {i+1} ---")
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
        
        out_path = f"output/audio/tests/f5_test_{i+1}.wav"
        sf.write(out_path, wav, sr)
        print(f"[+] Gerado em {latency:.2f}s | Duração: {audio_dur:.2f}s | RTF: {rtf:.2f}x | Salvo em: {out_path}")

if __name__ == "__main__":
    test_cloning()
