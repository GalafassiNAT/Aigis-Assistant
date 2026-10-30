import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import sounddevice as sd
import soundfile as sf
from piper.voice import PiperVoice

model_path = "models/piper/aigis_tts_ptbr.onnx"
config_path = "models/piper/aigis_tts_ptbr.onnx.json"
output_wav = "output/audio/teste_aigis_piper.wav"

print("============================================================")
print("TESTE E BENCHMARK PIPER ONNX DA AIGIS (DESTILACAO)")
print("============================================================")

print("[1/3] Carregando modelo Piper ONNX em RAM...")
t0 = time.perf_counter()
voice = PiperVoice.load(model_path, config_path=config_path)
sample_rate = voice.config.sample_rate
t_load = time.perf_counter() - t0
print(f"[OK] Modelo carregado em {t_load*1000:.1f}ms (Taxa de amostragem: {sample_rate}Hz)")

test_phrases = [
    "Olá, mestre! Meus sistemas vocais foram destilados com sucesso.",
    "Sim, mestre. Estou pronta para executar seus comandos.",
    "Entendido! Criando o script solicitado imediatamente.",
    "Prontidão confirmada. Aguardo sua próxima instrução.",
]

print("\n[2/3] Executando benchmark de latência (chamadas subsequentes)...")
for idx, text in enumerate(test_phrases, start=1):
    t0 = time.perf_counter()
    chunks = list(voice.synthesize(text))
    t_synth = time.perf_counter() - t0
    audio_float32 = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32) / 32768.0
    duration_s = len(audio_float32) / sample_rate
    rtf = t_synth / duration_s
    print(f"[{idx}/4] \"{text[:45]}...\"")
    print(f"      Tempo de síntese: {t_synth*1000:.1f}ms | Áudio: {duration_s:.2f}s | RTF: {rtf:.3f}x ({(1/rtf):.1f}x real-time)")

print("\n[3/3] Tocando frase 1 nos alto-falantes para validação de áudio...")
chunks = list(voice.synthesize(test_phrases[0]))
audio_float32 = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32) / 32768.0
sf.write(output_wav, audio_float32, sample_rate)
sd.play(audio_float32, sample_rate)
sd.wait()
print(f"[OK] Áudio de teste salvo em: {output_wav}")
print("============================================================")
