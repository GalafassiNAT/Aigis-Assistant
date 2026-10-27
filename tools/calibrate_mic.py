import time

import numpy as np
import sounddevice as sd

print("=" * 60)
print("🎙️ CALIBRADOR DE SENSIBILIDADE DO MICROFONE")
print("=" * 60)

# Usa o dispositivo padrão (Microfone Fuxi-H3)
sample_rate = 16000
chunk_size = int(sample_rate * 0.1)

print("\n1. Fique em SILÊNCIO por 3 segundos para medir o ruído ambiente...")
silence_rms = []
with sd.InputStream(
    samplerate=sample_rate, channels=1, dtype="float32", blocksize=chunk_size
) as stream:
    t_end = time.time() + 3.0
    while time.time() < t_end:
        data, _ = stream.read(chunk_size)
        rms = np.sqrt(np.mean(data**2))
        silence_rms.append(rms)

avg_silence = np.mean(silence_rms)
max_silence = np.max(silence_rms)
print(f"   Ruído ambiente médio: {avg_silence:.4f} | Pico: {max_silence:.4f}")

print("\n2. Agora FALE NORMALMENTE por 4 segundos ('Aigis, teste de áudio')...")
voice_rms = []
with sd.InputStream(
    samplerate=sample_rate, channels=1, dtype="float32", blocksize=chunk_size
) as stream:
    t_end = time.time() + 4.0
    while time.time() < t_end:
        data, _ = stream.read(chunk_size)
        rms = np.sqrt(np.mean(data**2))
        # Barra visual em tempo real
        bar = "█" * int(min(rms * 1000, 50))
        print(f"\rVolume: {rms:.4f} |{bar:<50}|", end="", flush=True)
        if rms > max_silence * 1.2:
            voice_rms.append(rms)

print("\n")
if voice_rms:
    avg_voice = np.mean(voice_rms)
    max_voice = np.max(voice_rms)
    # Ponto ideal de corte: um pouco acima do ruído máximo
    recommended_threshold = round((max_silence * 2 + avg_voice) / 3, 4)
    if recommended_threshold <= max_silence:
        recommended_threshold = round(max_silence * 1.5, 4)

    print("📊 RESULTADO DA CALIBRAÇÃO:")
    print(f"   - Ruído máximo em silêncio: {max_silence:.4f}")
    print(f"   - Volume médio da sua voz:  {avg_voice:.4f}")
    print(f"   - Pico da sua voz:          {max_voice:.4f}")
    print(f"   🎯 Limiar (Threshold) Recomendado: {recommended_threshold:.4f}")
else:
    print(
        "⚠️ Não detectamos voz acima do ruído. O microfone pode estar muito baixo nas configurações do Windows."
    )
