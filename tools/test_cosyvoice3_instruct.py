import sys
import os
import time
import soundfile as sf
import numpy as np
import torch

sys.path.append('tools/CosyVoice')
from cosyvoice.cli.cosyvoice import CosyVoice3

def test():
    print("[*] Carregando CosyVoice 3.0...")
    cosy = CosyVoice3('models/cosyvoice3/Fun-CosyVoice3-0.5B-2512')
    
    prompt_wav = 'models/voice_prompts/aigis_prompt_conversational.wav'
    with open('models/voice_prompts/aigis_prompt_conversational.txt', 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
        
    os.makedirs('output/audio/tests_cosyvoice3', exist_ok=True)
    
    # CosyVoice 3 exige instruct prompt com <|endofprompt|>
    instruct = "You are a helpful robotic assistant.<|endofprompt|>"
    
    tests = [
        ("cosy3_en_instruct", "Good evening. All systems are fully operational and ready for your command."),
        ("cosy3_pt_instruct", "Boa noite, mestre! Todos os sistemas estão operacionais e prontos para a sua ordem.")
    ]
    
    for tag, tts_text in tests:
        print(f"\n==========================================")
        print(f"[*] Testando {tag}: {tts_text}")
        t0 = time.perf_counter()
        
        # Testar inference_instruct2
        audio_chunks = []
        for model_output in cosy.inference_instruct2(tts_text, instruct, prompt_wav, stream=False):
            audio_chunks.append(model_output['tts_speech'].cpu().numpy().flatten())
            
        latency = time.perf_counter() - t0
        if audio_chunks:
            wav = np.concatenate(audio_chunks)
            sr = cosy.sample_rate
            dur = len(wav) / sr
            
            peak = np.max(np.abs(wav))
            if peak > 0:
                wav = (wav / peak) * 0.89
                
            out_p = f"output/audio/tests_cosyvoice3/{tag}.wav"
            sf.write(out_p, wav, sr)
            print(f"[+] Gerado em {latency:.2f}s | Duração: {dur:.2f}s | Sample Rate: {sr}Hz | Salvo: {out_p}")
        else:
            print("[!] Falha: Nenhum chunk de áudio gerado.")

if __name__ == '__main__':
    test()
