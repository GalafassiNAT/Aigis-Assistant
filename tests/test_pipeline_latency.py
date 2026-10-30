import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from core.voice import VoiceSynthesizer


async def bench():
    v = VoiceSynthesizer()
    phrases = [
        "Olá, mestre!",
        "Sistemas operacionais prontos e totalmente calibrados.",
        "Estou à sua disposição para qualquer tarefa técnica ou conversa hoje.",
    ]
    print("\n============================================================")
    print("MEDIÇÃO EMPÍRICA DE LATÊNCIA (EDGE-TTS + RVC DIRECTML EM RAM)")
    print("============================================================")
    for idx, p in enumerate(phrases, start=1):
        t0 = time.perf_counter()
        data, fs = await v.synthesize_array(p)
        dt = (time.perf_counter() - t0) * 1000
        dur_ms = (len(data) / fs) * 1000
        print(f"\n[{idx}/3] Frase: \"{p}\"")
        print(f"      Tempo de síntese: {dt:.0f}ms")
        print(f"      Duração da fala:  {dur_ms:.0f}ms")
        if dur_ms >= dt:
            print(f"      FOLGA NO BUFFER: +{dur_ms - dt:.0f}ms (A síntese termina ANTES da fala acabar!)")
        else:
            print(f"      Déficit: -{dt - dur_ms:.0f}ms")
    v.close()


if __name__ == "__main__":
    asyncio.run(bench())
