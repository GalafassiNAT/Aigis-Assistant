"""Pipeline de Treinamento e Exportação ONNX do Piper TTS para Google Colab com Backup Contínuo no Google Drive.

Instruções de Execução no Google Colab (GPU T4 ou L4):
1. Suba o arquivo 'dataset.zip' no Colab (ou coloque-o diretamente no seu Google Drive).
2. Execute este script:
   !git pull && python training/tts_distillation/3_colab_train_piper.py
3. Ao rodar, autorize a montagem do Google Drive quando solicitado na tela.
4. O script salvará tudo em tempo real na pasta 'Aigis_TTS_Model' do seu Google Drive.
"""

import os
import py_compile
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

BASE_CKPT = "pt_BR-faber-medium.ckpt"
ENV_DIR = "/content/piper_env"
ENV_PY = f"{ENV_DIR}/bin/python3" if os.path.exists("/content") else sys.executable
ENV_PIP = f"{ENV_DIR}/bin/pip" if os.path.exists("/content") else "pip"
DRIVE_BACKUP_DIR = Path("/content/drive/MyDrive/Aigis_TTS_Model")


def run_cmd(cmd: str):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Comando falhou com código {res.returncode}: {cmd}")


def mount_google_drive():
    """Monta o Google Drive de forma nativa e cria a pasta de persistência."""
    if os.path.exists("/content"):
        print("============================================================")
        print("💾 CONECTANDO AO GOOGLE DRIVE (PERSISTÊNCIA GARANTIDA)")
        print("============================================================")
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            DRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            print(f"✅ Google Drive conectado com sucesso! Destino: {DRIVE_BACKUP_DIR}")
        except Exception as e:
            print(f"⚠️ Aviso ao conectar Google Drive: {e}")


def patch_piper_sources():
    """Restaura arquivos originais do repositório piper e aplica patches seguros e de backup contínuo."""
    run_cmd("git -C piper checkout .")

    # 1. Patch seguro de add_audio em vits/lightning.py (bloco completo de 3 linhas)
    lightning_path = Path("piper/src/python/piper_train/vits/lightning.py")
    content = lightning_path.read_text(encoding="utf-8")
    old_audio_block = (
        "            self.logger.experiment.add_audio(\n"
        "                tag, test_audio, sample_rate=self.hparams.sample_rate\n"
        "            )"
    )
    new_audio_block = (
        "            if hasattr(self.logger.experiment, 'add_audio'):\n"
        "                self.logger.experiment.add_audio(\n"
        "                    tag, test_audio, sample_rate=self.hparams.sample_rate\n"
        "                )"
    )
    if old_audio_block in content:
        content = content.replace(old_audio_block, new_audio_block, 1)
        lightning_path.write_text(content, encoding="utf-8")

    # 2. Patch em __main__.py: ModelCheckpoint + DriveSyncCallback para salvar cada época no Google Drive
    main_path = Path("piper/src/python/piper_train/__main__.py")
    main_content = main_path.read_text(encoding="utf-8")

    callback_code = (
        "from pytorch_lightning.callbacks import Callback, ModelCheckpoint\n"
        "import shutil, glob\n"
        "\n"
        "class DriveSyncCallback(Callback):\n"
        "    def on_train_epoch_end(self, trainer, pl_module):\n"
        "        try:\n"
        "            target_dir = '/content/drive/MyDrive/Aigis_TTS_Model'\n"
        "            if os.path.exists('/content/drive/MyDrive'):\n"
        "                os.makedirs(target_dir, exist_ok=True)\n"
        "                root = str(trainer.default_root_dir or 'training_output')\n"
        "                for ckpt in glob.glob(f'{root}/**/*.ckpt', recursive=True):\n"
        "                    shutil.copy(ckpt, target_dir)\n"
        "        except Exception:\n"
        "            pass\n"
    )

    old_ckpt_line = "trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)]"
    new_ckpt_line = (
        "trainer.callbacks = [\n"
        "        ModelCheckpoint(every_n_epochs=args.checkpoint_epochs, save_last=True, save_top_k=1),\n"
        "        DriveSyncCallback()\n"
        "    ]"
    )

    if old_ckpt_line in main_content:
        main_content = callback_code + "\n" + main_content.replace(old_ckpt_line, new_ckpt_line, 1)
        main_path.write_text(main_content, encoding="utf-8")

    # Valida sintaxe dos arquivos modificados antes de prosseguir
    py_compile.compile(str(lightning_path), doraise=True)
    py_compile.compile(str(main_path), doraise=True)
    print("[PATCH] Código-fonte do piper_train com sincronização contínua para o Drive validado.")


def setup_colab_environment():
    mount_google_drive()

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

    # 4. Instala piper-phonemize e dependências cravadas para evitar conflitos de versão
    wheel_url = (
        "https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/"
        "piper_phonemize-1.1.0-cp310-cp310-manylinux_2_28_x86_64.whl"
    )
    run_cmd(f"{ENV_PIP} install '{wheel_url}'")
    run_cmd(
        f'{ENV_PIP} install "numpy<2.0" piper-tts onnx onnxruntime '
        f"pytorch-lightning==1.9.5 torchmetrics==0.11.4 tensorboard cython librosa==0.9.2"
    )

    # 5. Clona e restaura piper/src/python, aplica patches e compila monotonic_align
    if not os.path.exists("piper"):
        run_cmd("git clone https://github.com/rhasspy/piper.git")
    patch_piper_sources()

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

    # 6. Descompacta e normaliza estrutura do dataset (busca local ou no Google Drive)
    if not os.path.exists("dataset.zip"):
        for drive_candidate in [
            DRIVE_BACKUP_DIR / "dataset.zip",
            Path("/content/drive/MyDrive/dataset.zip"),
        ]:
            if drive_candidate.exists():
                print(f"[DATASET] Encontrado no Google Drive: {drive_candidate}. Copiando...")
                shutil.copy(str(drive_candidate), "dataset.zip")
                break

    if os.path.exists("dataset.zip") and not os.path.exists("dataset/metadata.csv"):
        print("[DATASET] Descompactando dataset.zip...")
        with zipfile.ZipFile("dataset.zip", "r") as z:
            z.extractall(".")

        nested_dir = Path("dataset/dataset")
        if nested_dir.exists() and (nested_dir / "metadata.csv").exists():
            for item in nested_dir.iterdir():
                dest = Path("dataset") / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(nested_dir, ignore_errors=True)

    assert os.path.exists("dataset/metadata.csv"), (
        "Arquivo dataset.zip não encontrado! Suba 'dataset.zip' no Colab ou coloque na pasta "
        "'Aigis_TTS_Model' do seu Google Drive."
    )
    wav_count = len(list(Path("dataset/wavs").glob("*.wav")))
    print(f"[DATASET] Pronto com {wav_count} arquivos de áudio em dataset/wavs/ e metadata.csv.")

    # 7. Baixa checkpoint base Faber Medium (1.2M steps, single-speaker PT-BR)
    if not os.path.exists(BASE_CKPT):
        print("[CHECKPOINT] Baixando checkpoint base Faber Medium pré-treinado...")
        ckpt_url = (
            "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
            "pt/pt_BR/faber/medium/epoch%3D6159-step%3D1230728.ckpt"
        )
        run_cmd(f"wget -q -O {BASE_CKPT} '{ckpt_url}'")


def get_checkpoint_epoch(ckpt_path: str) -> int:
    """Lê a época atual gravada dentro do checkpoint do PyTorch Lightning."""
    cmd = (
        f"{ENV_PY} -c \"import torch; "
        f"ckpt = torch.load('{ckpt_path}', map_location='cpu', weights_only=False); "
        f"print(int(ckpt.get('epoch', 0)))\""
    )
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    return int(out.splitlines()[-1])


def train_and_export(
    dataset_dir: str = "dataset",
    output_dir: str = "training_output",
    finetune_epochs: int = 200,
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

    # 2. Pre-flight check de exportação ONNX antes de iniciar o treino
    print("[PRE-FLIGHT] Validando pipeline de exportação ONNX no checkpoint base...")
    run_cmd(f"{ENV_PY} -m piper_train.export_onnx {BASE_CKPT} /tmp/preflight_test.onnx")
    if os.path.exists("/tmp/preflight_test.onnx"):
        os.remove("/tmp/preflight_test.onnx")
    print("✅ Pre-flight ONNX validado com sucesso!")

    # 3. Calcula max_epochs somando as épocas de fine-tuning à época do checkpoint base (6159)
    base_epoch = get_checkpoint_epoch(BASE_CKPT)
    target_max_epochs = base_epoch + finetune_epochs
    print(
        f"[INFO] Checkpoint base está na época {base_epoch}. "
        f"Treinando por +{finetune_epochs} épocas com batch_size={batch_size} (até max_epochs={target_max_epochs})..."
    )

    # 4. Execução do Treinamento Single-Speaker com sincronização para o Drive
    train_cmd = (
        f"{ENV_PY} -m piper_train "
        f"--dataset-dir {output_dir} "
        f"--accelerator gpu "
        f"--devices 1 "
        f"--batch-size {batch_size} "
        f"--max_epochs {target_max_epochs} "
        f"--checkpoint-epochs 10 "
        f"--precision 32 "
        f"--quality medium "
        f"--resume_from_checkpoint {BASE_CKPT}"
    )
    run_cmd(train_cmd)

    # 5. Identifica o último checkpoint salvo
    lightning_logs = Path(output_dir) / "lightning_logs"
    ckpts = list(lightning_logs.glob("**/*.ckpt"))
    if not ckpts:
        raise RuntimeError("Nenhum checkpoint encontrado em training_output/lightning_logs.")
    best_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"\n[EXPORT] Checkpoint final selecionado: {best_ckpt}")

    # 6. Exportação para ONNX
    export_onnx_cmd = f"{ENV_PY} -m piper_train.export_onnx {best_ckpt} aigis_tts_ptbr.onnx"
    run_cmd(export_onnx_cmd)

    # 7. Copia o arquivo config.json para aigis_tts_ptbr.onnx.json
    config_src = Path(output_dir) / "config.json"
    if config_src.exists():
        shutil.copy(str(config_src), "aigis_tts_ptbr.onnx.json")

    # 8. Backup garantido e redundante no Google Drive
    if Path("/content/drive/MyDrive").exists():
        print(f"\n[BACKUP] Salvando modelo final no seu Google Drive ({DRIVE_BACKUP_DIR})...")
        DRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy("aigis_tts_ptbr.onnx", str(DRIVE_BACKUP_DIR / "aigis_tts_ptbr.onnx"))
        shutil.copy("aigis_tts_ptbr.onnx.json", str(DRIVE_BACKUP_DIR / "aigis_tts_ptbr.onnx.json"))
        shutil.copy(str(best_ckpt), str(DRIVE_BACKUP_DIR / "aigis_best_checkpoint.ckpt"))
        print(f"✅ ARQUIVOS SALVOS NO GOOGLE DRIVE COM SUCESSO: {DRIVE_BACKUP_DIR}")

    # 9. Disparo automático de download no navegador
    try:
        from google.colab import files
        print("[DOWNLOAD] Disparando download automático no navegador...")
        files.download("aigis_tts_ptbr.onnx")
        files.download("aigis_tts_ptbr.onnx.json")
    except Exception:
        pass

    print("\n============================================================")
    print("✅ MODELO ONNX EXPORTADO E SALVO COM SUCESSO TOTAL!")
    print(f"Arquivos salvos no Google Drive em: {DRIVE_BACKUP_DIR}")
    print("- aigis_tts_ptbr.onnx")
    print("- aigis_tts_ptbr.onnx.json")
    print("============================================================")


if __name__ == "__main__":
    setup_colab_environment()
    train_and_export()
