"""Script para Google Colab: Teste Zero-Shot de Alta Fidelidade (XTTS-v2 e F5-TTS).

Objetivo:
Gerar amostras de áudio em Português (PT-BR) clonando diretamente a voz oficial da Aigis
a partir do áudio limpo de estúdio do jogo Persona 3 Reload ('aigis_original_sample.wav').

Como usar no Colab (GPU T4):
1. Suba o arquivo 'output/audio/aigis_original_sample.wav' para o Colab.
2. Execute:
   !python training/test_voice_cloners_colab.py
3. Ouça os arquivos gerados direto no Colab:
   - 'aigis_xtts_ptbr.wav'
   - 'aigis_f5_ptbr.wav'
"""

import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Comando falhou com código {res.returncode}: {cmd}")


REF_AUDIO = "aigis_original_sample.wav"
REF_TEXT = (
    "The fact that a future exists is not a guarantee that all will have access to it. "
    "It's something so obvious, and yet we did not understand."
)
TARGET_TEXT = "Olá, mestre! Sistemas operacionais prontos e totalmente calibrados. Como posso ajudar você hoje?"


def test_xtts_v2():
    print("============================================================")
    print("🎙️ TESTE 1: XTTS-v2 (COQUI) - CLONAGEM ZERO-SHOT EM PT-BR")
    print("============================================================")
    run_cmd("pip install -q TTS soundfile")

    from TTS.api import TTS
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[XTTS] Carregando modelo XTTS-v2 no dispositivo: {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    output_path = "aigis_xtts_ptbr.wav"
    print(f"[XTTS] Sintetizando em Português com a voz da Aigis: \"{TARGET_TEXT}\"...")
    tts.tts_to_file(
        text=TARGET_TEXT,
        speaker_wav=REF_AUDIO,
        language="pt",
        file_path=output_path,
        split_sentences=True,
    )
    print(f"✅ Áudio XTTS-v2 gerado com sucesso: '{output_path}'")


def test_f5_tts():
    print("\n============================================================")
    print("🎙️ TESTE 2: F5-TTS (FLOW-MATCHING DE ÚLTIMA GERAÇÃO)")
    print("============================================================")
    run_cmd("pip install -q f5-tts soundfile")

    from f5_tts.api import F5TTS

    print("[F5-TTS] Carregando pipeline F5-TTS...")
    f5 = F5TTS()

    output_path = "aigis_f5_ptbr.wav"
    print(f"[F5-TTS] Sintetizando com referência cruzada (EN -> PT): \"{TARGET_TEXT}\"...")
    wav, sr, _ = f5.infer(
        ref_file=REF_AUDIO,
        ref_text=REF_TEXT,
        gen_text=TARGET_TEXT,
        file_wave=output_path,
    )
    print(f"✅ Áudio F5-TTS gerado com sucesso: '{output_path}'")


if __name__ == "__main__":
    if not os.path.exists(REF_AUDIO):
        print(f"ERRO: Arquivo '{REF_AUDIO}' não encontrado no diretório atual!")
        print("Faça o upload do arquivo 'output/audio/aigis_original_sample.wav' para a raiz do Colab.")
        sys.exit(1)

    test_xtts_v2()
    try:
        test_f5_tts()
    except Exception as e:
        print(f"Aviso no teste F5-TTS: {e}")

    print("\n============================================================")
    print("🎉 TESTES CONCLUÍDOS!")
    print("Baixe ou reproduza os áudios gerados no Colab:")
    print("1. aigis_xtts_ptbr.wav (XTTS-v2)")
    print("2. aigis_f5_ptbr.wav (F5-TTS)")
    print("============================================================")
