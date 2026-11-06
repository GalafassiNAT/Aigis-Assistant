"""
Test Portuguese phonetic transcription mapped to English spelling rules for F5-TTS.
Since F5-TTS has 99.96% precision on English text with Aigis's voice, mapping PT-BR
phonemes into English graphemes bypasses the English accent!
"""
import time
import os
import soundfile as sf
import numpy as np
import torch
from f5_tts.api import F5TTS

def test_pt_anglicized():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Carregando F5-TTS no dispositivo: {device}...")
    f5 = F5TTS(device=device)
    
    ref_audio = "models/voice_prompts/aigis_prompt_conversational.wav"
    with open("models/voice_prompts/aigis_prompt_conversational.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
        
    os.makedirs("output/audio/tests_pt_phonetic", exist_ok=True)
    
    # Frase original: "Boa noite, mestre! Todos os sistemas estão operacionais e prontos para a sua ordem."
    # Transcrição fonética adaptada para grafemas ingleses que o F5-TTS lê sem sotaque:
    candidates = [
        (
            "pt_map_1",
            "Boh-ah noy-chee, meh-stree! Toh-doos oos seess-teh-mas ess-town oh-peh-rah-see-oh-nah-ees, ee prohn-toos pah-rah ah soo-ah ohr-deng."
        ),
        (
            "pt_map_2",
            "Boh-ah noy-te, mess-tree! Toh-dos os sis-teh-mas ess-tahm oh-peh-rah-see-oh-nice, ee prohn-tos pah-rah ah soo-ah or-dem."
        ),
        (
            "pt_curto",
            "Boh-ah noy-chee, mess-tree. See-steh-mas oh-peh-rah-see-oh-nah-ees prohn-toos."
        )
    ]
    
    for tag, text in candidates:
        print(f"\n==========================================")
        print(f"[*] Testando {tag}: \"{text}\"")
        t0 = time.perf_counter()
        
        wav, sr, _ = f5.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            nfe_step=32,
            speed=0.88,
            target_rms=0.08
        )
        latency = time.perf_counter() - t0
        audio_dur = len(wav) / sr
        
        peak = np.max(np.abs(wav))
        wav_norm = (wav / peak) * 0.89 if peak > 0 else wav
        
        out_p = f"output/audio/tests_pt_phonetic/{tag}.wav"
        sf.write(out_p, wav_norm, sr)
        print(f"[+] Gerado em {latency:.2f}s | Duração: {audio_dur:.2f}s | Salvo: {out_p}")

if __name__ == "__main__":
    test_pt_anglicized()
