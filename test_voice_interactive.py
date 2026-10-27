# -*- coding: utf-8 -*-
import asyncio
import sys
from pathlib import Path

# Garante importação da raiz
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.voice import VoiceSynthesizer


async def main():
    print("==================================================")
    print("   PROJETO AIGIS - SÍNTESE E REPRODUÇÃO LOCAL    ")
    print("==================================================")
    
    synthesizer = VoiceSynthesizer()
    print("\nDigite uma frase para a Aigis falar (ou 'sair' para encerrar):\n")

    try:
        while True:
            try:
                texto = input("Você > ").strip()
                if not texto:
                    continue
                if texto.lower() in ["sair", "exit", "quit"]:
                    print("Encerrando teste de voz.")
                    break

                print("[Aigis sintetizando...]")
                output_file = await synthesizer.speak(texto, output_filename="aigis_live.wav", play_audio=True)
                print(f"[Áudio reproduzido! Salvo em: {output_file}]\n")
            except (KeyboardInterrupt, EOFError):
                print("\nEncerrando teste.")
                break
            except Exception as e:
                print(f"[Erro]: {e}\n")
    finally:
        synthesizer.close()

if __name__ == "__main__":
    asyncio.run(main())
