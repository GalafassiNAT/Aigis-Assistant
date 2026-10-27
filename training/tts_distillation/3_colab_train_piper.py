"""Pipeline de Treinamento e Exportação ONNX do Piper TTS para Google Colab.

Instruções de Execução no Google Colab (GPU T4 ou L4):
1. Suba o arquivo 'dataset.zip' (gerado a partir de dataset/ com wavs/ e metadata.csv).
2. Execute este script:
   !python 3_colab_train_piper.py
3. Baixe o modelo final exportado: 'aigis_tts_ptbr.onnx' e 'aigis_tts_ptbr.onnx.json'.
4. Coloque-os na pasta 'models/piper/' do projeto Aigis.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def run_cmd(cmd: str):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Comando falhou com código {res.returncode}: {cmd}")


def setup_colab_environment():
    print("============================================================")
    print("🚀 CONFIGURANDO AMBIENTE PIPER TTS NO COLAB")
    print("============================================================")

    # 1. Instala dependências de áudio e fonetizador espeak-ng
    run_cmd("apt-get update && apt-get install -y espeak-ng libespeak-ng-dev")
    run_cmd("pip install piper-tts piper-phonemize onnx onnxruntime torch pytorch-lightning==1.9.5")

    # 2. Clona o repositório de treino do Piper se necessário
    if not os.path.exists("piper"):
        run_cmd("git clone https://github.com/rhasspy/piper.git")
        run_cmd("pip install -e piper/src/python")

    # 3. Descompacta dataset se houver dataset.zip
    if os.path.exists("dataset.zip") and not os.path.exists("dataset/metadata.csv"):
        print("[DATASET] Descompactando dataset.zip...")
        with zipfile.ZipFile("dataset.zip", "r") as z:
            z.extractall("dataset")

    # 4. Baixa checkpoint base pré-treinado em Português (Faber Medium)
    base_ckpt = "pt_BR-faber-medium.ckpt"
    if not os.path.exists(base_ckpt):
        print("[CHECKPOINT] Baixando checkpoint base pt_BR-faber-medium...")
        url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"
        # Download do checkpoint PyTorch de fine-tuning
        ckpt_url = "https://github.com/rhasspy/piper/releases/download/v0.0.2/pt_BR-faber-medium.ckpt"
        try:
            run_cmd(f"wget -q -O {base_ckpt} {ckpt_url}")
        except Exception:
            print("[INFO] Checkpoint inicial de download direto indisponível, inicializando treinamento single-speaker.")


def train_and_export(
    dataset_dir: str = "dataset",
    output_dir: str = "training_output",
    max_epochs: int = 1500,
    batch_size: int = 32,
):
    print("============================================================")
    print("🏋️ INICIANDO PRÉ-PROCESSAMENTO E TREINO PIPER (VITS)")
    print("============================================================")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Pré-processamento Fonético LJSpeech
    preprocess_cmd = (
        f"python3 -m piper_train.preprocess "
        f"--language pt-br "
        f"--input-dir {dataset_dir} "
        f"--output-dir {output_dir} "
        f"--dataset-format ljspeech "
        f"--single-speaker "
        f"--sample-rate 22050"
    )
    run_cmd(preprocess_cmd)

    # 2. Execução do Treinamento
    train_cmd = (
        f"python3 -m piper_train "
        f"--dataset-dir {output_dir} "
        f"--accelerator gpu "
        f"--devices 1 "
        f"--batch-size {batch_size} "
        f"--max_epochs {max_epochs} "
        f"--checkpoint-epochs 100 "
        f"--precision 32"
    )
    if os.path.exists("pt_BR-faber-medium.ckpt"):
        train_cmd += " --resume_from_single_speaker_checkpoint pt_BR-faber-medium.ckpt"

    run_cmd(train_cmd)

    # 3. Identifica o último checkpoint salvo
    lightning_logs = Path(output_dir) / "lightning_logs"
    ckpts = list(lightning_logs.glob("**/*.ckpt"))
    if not ckpts:
        raise RuntimeError("Nenhum checkpoint encontrado para exportação.")
    best_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"\n[EXPORT] Melhor checkpoint encontrado: {best_ckpt}")

    # 4. Exportação para ONNX
    export_onnx_cmd = f"python3 -m piper_train.export_onnx {best_ckpt} aigis_tts_ptbr.onnx"
    run_cmd(export_onnx_cmd)

    print("\n============================================================")
    print("✅ MODELO ONNX EXPORTADO COM SUCESSO!")
    print("Arquivos prontos para download:")
    print("- aigis_tts_ptbr.onnx")
    print("- aigis_tts_ptbr.onnx.json")
    print("============================================================")


if __name__ == "__main__":
    setup_colab_environment()
    train_and_export()
