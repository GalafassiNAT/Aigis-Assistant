"""
Test CosyVoice 3.0 Zero-Shot Cross-Lingual with Aigis Tartarus prompt.
1. English target phrase (benchmark quality)
2. Portuguese target phrase (cross-lingual phonetic assessment)
"""
import sys
import os
import time
import soundfile as sf
import numpy as np
import torch

sys.path.append('tools/CosyVoice')
from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import load_wav

def test_cosyvoice3():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Inicializando CosyVoice 3.0 no dispositivo: {device}...")
    cosy = CosyVoice3('models/cosyvoice3/Fun-CosyVoice3-0.5B-2512')
    
    prompt_wav = 'models/voice_prompts/aigis_prompt_conversational.wav'
    with open('models/voice_prompts/aigis_prompt_conversational.txt', 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
        
    print(f"[*] Prompt de referência: \"{prompt_text}\" ({prompt_wav})")
    os.makedirs('output/audio/tests_cosyvoice3', exist_ok=True)
    
    tests = [
        ("cosy3_en", "Good evening. All systems are fully operational and ready for your command.", prompt_text),
        ("cosy3_pt_cross", "Boa noite, mestre! Todos os sistemas estão operacionais e prontos para a sua ordem.", prompt_text),
    ]
    
    for tag, tts_text, p_text in tests:
        print(f"\n==========================================")
        print(f"[*] Testando {tag}: \"{tts_text}\"")
        t0 = time.perf_counter()
        
        # Testar cross-lingual zero-shot nativo do CosyVoice 3
        # No CosyVoice 3, frontend_cross_lingual ou inference_cross_lingual transfere o timbre da língua A para a língua B
        audio_chunks = []
        for chunk in cosy.inference_cross_lingual(tts_text, prompt_wav, stream=False):
            audio_chunks.append(chunk['tts_speech'].numpy().flatten())
            
        latency = time.perf_counter() - t0
        if audio_chunks:
            wav = np.concatenate(audio_chunks)
            sr = cosy.sample_rate
            dur = len(wav) / sr
            
            # Normalizar para -1.0 dBFS
            peak = np.max(np.abs(wav))
            if peak > 0:
                wav = (wav / peak) * 0.89
                
            out_p = f"output/audio/tests_cosyvoice3/{tag}.wav"
            sf.write(out_p, wav, sr)
            print(f"[+] Gerado em {latency:.2f}s | Duração: {dur:.2f}s | Sample Rate: {sr}Hz | Salvo: {out_p}")
        else:
            print(f"[!] Falha: Nenhum chunk de áudio gerado para {tag}")

if __name__ == '__main__':
    test_cosyvoice3()
