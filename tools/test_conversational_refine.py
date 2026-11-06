"""
Refine Conversational (speed=0.88, target_rms=0.08) and test Portuguese Phonetic Adaptation.
"""
import time
import os
import soundfile as sf
import numpy as np
import torch
from f5_tts.api import F5TTS

def test_refinement():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Carregando F5-TTS no dispositivo: {device}...")
    f5 = F5TTS(device=device)
    
    ref_audio = "models/voice_prompts/aigis_prompt_conversational.wav"
    with open("models/voice_prompts/aigis_prompt_conversational.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
        
    os.makedirs("output/audio/tests_refinement", exist_ok=True)
    
    # 1. Ajuste fino de cadência na frase em inglês (speed=0.88 para tirar a aceleração)
    print("\n--- 1. Inglês Conversational Refinado (speed=0.88) ---")
    wav_en, sr, _ = f5.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text="Good evening. All systems are fully operational and ready for your command.",
        nfe_step=32,
        speed=0.88,
        target_rms=0.08
    )
    peak = np.max(np.abs(wav_en))
    wav_en = (wav_en / peak) * 0.89 if peak > 0 else wav_en
    sf.write("output/audio/tests_refinement/conversational_speed088.wav", wav_en, sr)
    print("Salvo: output/audio/tests_refinement/conversational_speed088.wav")

    # 2. Teste em Português com a mesma referência conversacional
    # Texto direto em PT-BR vs Texto com guia fonético suave
    pt_tests = [
        ("pt_direto", "Boa noite, mestre. Todos os sistemas estão operacionais e prontos para a sua ordem."),
        ("pt_fonetico", "Boa noite, mestre. Todos os sistemas estão operacionáis, e prontos para a sua ordem.")
    ]
    
    for tag, text in pt_tests:
        print(f"\n--- 2. Português Conversational ({tag}) ---")
        wav_pt, sr, _ = f5.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            nfe_step=32,
            speed=0.88,
            target_rms=0.08
        )
        peak = np.max(np.abs(wav_pt))
        wav_pt = (wav_pt / peak) * 0.89 if peak > 0 else wav_pt
        out_p = f"output/audio/tests_refinement/{tag}.wav"
        sf.write(out_p, wav_pt, sr)
        print(f"Salvo: {out_p}")

if __name__ == "__main__":
    test_refinement()
