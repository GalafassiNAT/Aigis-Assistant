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

BASE_CKPT = "pt_BR-faber-medium.ckpt"
ENV_DIR = "/content/piper_env"
ENV_PY = f"{ENV_DIR}/bin/python3" if os.path.exists("/content") else sys.executable
ENV_PIP = f"{ENV_DIR}/bin/pip" if os.path.exists("/content") else "pip"


def run_cmd(cmd: str):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Comando falhou com código {res.returncode}: {cmd}")


def setup_colab_environment():
    print("============================================================")
    print("🚀 CONFIGURANDO AMBIENTE PIPER TTS NO COLAB (Python 3.10)")
    print("============================================================")

    # 1. Instala Python 3.10 e espeak-ng no Ubuntu 24.04 do Colab
    run_cmd("apt-get update && apt-get install -y espeak-ng libespeak-ng-dev python3.10 python3.10-venv python3.10-dev")

    # 2. Cria ambiente virtual com Python 3.10 compatível com piper-phonemize
    if os.path.exists("/content") and not os.path.exists(ENV_DIR):
        run_cmd(f"python3.10 -m venv {ENV_DIR}")

    # 3. Usa pip compatível (<=24.0) e instala PyTorch com suporte a CUDA
    run_cmd(f"{ENV_PIP} install pip==24.0")
    run_cmd(f"{ENV_PIP} install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")

    # 4. Instala piper-phonemize direto da wheel oficial para Python 3.10
    wheel_url = (
        "https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/"
        "piper_phonemize-1.1.0-cp310-cp310-manylinux_2_28_x86_64.whl"
    )
    run_cmd(f"{ENV_PIP} install '{wheel_url}'")
    run_cmd(f"{ENV_PIP} install piper-tts onnx onnxruntime onnxsim pytorch-lightning==1.9.5 cython librosa")

    # 5. Clona e instala piper/src/python no ambiente e compila monotonic_align
    if not os.path.exists("piper"):
        run_cmd("git clone https://github.com/rhasspy/piper.git")
    run_cmd(f"{ENV_PIP} install --no-deps -e piper/src/python")
    print("[BUILD] Compilando extensão Cython monotonic_align...")
    align_dir = "piper/src/python/piper_train/vits/monotonic_align"
    run_cmd(f"cd {align_dir} && mkdir -p monotonic_align && {ENV_DIR}/bin/cythonize -i core.pyx")
    run_cmd(f"cd {align_dir} && cp -f core*.so monotonic_align/ 2>/dev/null || true")
    run_cmd(f"cd {align_dir} && {ENV_PY} setup.py build_ext --inplace 2>/dev/null || true")
    run_cmd(f"cd {align_dir} && cp -f core*.so monotonic_align/ 2>/dev/null || true")

    # Validação imediata de import do monotonic_align
    test_cmd = (
        f"{ENV_PY} -c 'from piper_train.vits.monotonic_align import maximum_path; "
        f"print(\"✅ Monotonic align compilado e verificado com sucesso!\")'"
    )
    run_cmd(test_cmd)

    # 6. Descompacta e normaliza estrutura do dataset
    if os.path.exists("dataset.zip"):
        print("[DATASET] Descompactando dataset.zip...")
        with zipfile.ZipFile("dataset.zip", "r") as z:
            z.extractall(".")

        # Corrige aninhamento se foi compactado como dataset/dataset
        nested_dir = Path("dataset/dataset")
        if nested_dir.exists() and (nested_dir / "metadata.csv").exists():
            import shutil
            for item in nested_dir.iterdir():
                dest = Path("dataset") / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(nested_dir, ignore_errors=True)

    assert os.path.exists("dataset/metadata.csv"), "Arquivo dataset/metadata.csv não encontrado!"
    wav_count = len(list(Path("dataset/wavs").glob("*.wav")))
    print(f"[DATASET] Pronto com {wav_count} arquivos de áudio em dataset/wavs/ e metadata.csv.")

    # 7. Baixa checkpoint base Faber Medium (1.2M steps)
    if not os.path.exists(BASE_CKPT):
        print("[CHECKPOINT] Baixando checkpoint base Faber Medium pré-treinado...")
        ckpt_url = (
            "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
            "pt/pt_BR/faber/medium/epoch%3D6159-step%3D1230728.ckpt"
        )
        run_cmd(f"wget -q -O {BASE_CKPT} '{ckpt_url}'")


def train_and_export(
    dataset_dir: str = "dataset",
    output_dir: str = "training_output",
    max_epochs: int = 300,
    batch_size: int = 32,
):
    print("============================================================")
    print("🏋️ INICIANDO PRÉ-PROCESSAMENTO E TREINO PIPER (VITS)")
    print("============================================================")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Pré-processamento Fonético LJSpeech (se ainda não realizado)
    dataset_jsonl = Path(output_dir) / "dataset.jsonl"
    if not dataset_jsonl.exists():
        preprocess_cmd = (
            f"{ENV_PY} -m piper_train.preprocess "
            f"--language pt-br "
            f"--input-dir {dataset_dir} "
            f"--output-dir {output_dir} "
            f"--dataset-format ljspeech "
            f"--single-speaker "
            f"--sample-rate 22050"
        )
        run_cmd(preprocess_cmd)
    else:
        print("[INFO] Pré-processamento já existente em training_output/dataset.jsonl. Prosseguindo...")

    # 2. Execução do Treinamento
    train_cmd = (
        f"{ENV_PY} -m piper_train "
        f"--dataset-dir {output_dir} "
        f"--accelerator gpu "
        f"--devices 1 "
        f"--batch-size {batch_size} "
        f"--max_epochs {max_epochs} "
        f"--checkpoint-epochs 25 "
        f"--precision 32 "
        f"--quality medium"
    )
    if os.path.exists(BASE_CKPT):
        train_cmd += f" --resume_from_single_speaker_checkpoint {BASE_CKPT}"

    run_cmd(train_cmd)

    # 3. Identifica o último checkpoint salvo
    lightning_logs = Path(output_dir) / "lightning_logs"
    ckpts = list(lightning_logs.glob("**/*.ckpt"))
    if not ckpts:
        raise RuntimeError("Nenhum checkpoint encontrado para exportação.")
    best_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"\n[EXPORT] Melhor checkpoint encontrado: {best_ckpt}")

    # 4. Exportação para ONNX
    export_onnx_cmd = f"{ENV_PY} -m piper_train.export_onnx {best_ckpt} aigis_tts_ptbr.onnx"
    run_cmd(export_onnx_cmd)

    # 5. Copia o arquivo config.json para aigis_tts_ptbr.onnx.json
    config_src = Path(output_dir) / "config.json"
    if config_src.exists():
        import shutil
        shutil.copy(str(config_src), "aigis_tts_ptbr.onnx.json")

    print("\n============================================================")
    print("✅ MODELO ONNX EXPORTADO COM SUCESSO!")
    print("Arquivos prontos para download:")
    print("- aigis_tts_ptbr.onnx")
    print("- aigis_tts_ptbr.onnx.json")
    print("============================================================")


if __name__ == "__main__":
    setup_colab_environment()
    train_and_export()
