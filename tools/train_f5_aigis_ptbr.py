"""Pipeline de Fine-Tuning do F5-TTS v1 Base para Aigis (PT-BR).

Executa o treinamento na GPU AMD Radeon RX 7600 sob ROCm 6.2.
Ajusta os pesos do DiT a partir do checkpoint oficial `F5TTS_v1_Base`
com o dataset pareado de alta fidelidade em português brasileiro e falas
autênticas da Dawn Bennett.
"""

import argparse
import json
import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path

import torch
from datasets import Dataset as Dataset_
from cached_path import cached_path

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import CustomDataset
from f5_tts.model.utils import get_tokenizer

# Configuração de Hiperparâmetros de Áudio
TARGET_SAMPLE_RATE = 24000
N_MEL_CHANNELS = 100
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
MEL_SPEC_TYPE = "vocos"


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento Fine-Tuning F5-TTS PT-BR")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Taxa de aprendizado")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamanho do lote por GPU")
    parser.add_argument("--grad_accum", type=int, default=8, help="Passos de acumulação de gradiente")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas")
    parser.add_argument("--save_per_updates", type=int, default=200, help="Salvar a cada N passos")
    parser.add_argument("--last_per_updates", type=int, default=50, help="Salvar last checkpoint a cada N passos")
    parser.add_argument("--ckpt_dir", type=str, default="training/f5_ptbr/ckpts", help="Diretório dos checkpoints")
    parser.add_argument("--prepped_dir", type=str, default="training/f5_ptbr/prepped", help="Diretório dos dados pré-processados")
    parser.add_argument("--max_updates", type=int, default=None, help="Parar após N updates para testes rápidos")
    return parser.parse_args()


def main():
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    ckpt_dir = project_root / args.ckpt_dir
    prepped_dir = project_root / args.prepped_dir
    raw_arrow_path = prepped_dir / "raw.arrow"
    dur_json_path = prepped_dir / "duration.json"
    vocab_path = prepped_dir / "vocab.txt"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("🚀 INICIANDO FINE-TUNING DO F5-TTS (AIGIS PT-BR)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Dados Arrow: {raw_arrow_path}")
    print("==================================================================")

    # 1. Carrega vocabulário
    if not vocab_path.exists():
        vocab_path = Path(str(files("f5_tts").joinpath("infer/examples/vocab.txt")))
    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    print(f"Tamanho do vocabulário: {vocab_size} caracteres")

    # 2. Configura o modelo DiT e CFM
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        checkpoint_activations=True,
    )

    mel_spec_kwargs = dict(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mel_channels=N_MEL_CHANNELS,
        target_sample_rate=TARGET_SAMPLE_RATE,
        mel_spec_type=MEL_SPEC_TYPE,
    )

    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=N_MEL_CHANNELS),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # 3. Prepara checkpoint base para fine-tuning
    target_pretrained_file = ckpt_dir / "pretrained_model_1250000.safetensors"
    if not any(ckpt_dir.glob("model_*.pt")) and not target_pretrained_file.exists():
        print("\nBaixando/Verificando checkpoint base oficial F5TTS_v1_Base...")
        base_ckpt = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
        shutil.copy2(base_ckpt, target_pretrained_file)
        print(f"Copiado para: {target_pretrained_file}")

    # 4. Carrega o dataset a partir do Arrow pré-compilado
    print("\nCarregando dataset do Arrow...")
    arrow_ds = Dataset_.from_file(str(raw_arrow_path))
    with open(dur_json_path, "r", encoding="utf-8") as f:
        durations = json.load(f)["duration"]

    train_dataset = CustomDataset(
        arrow_ds,
        durations=durations,
        preprocessed_mel=False,
        **mel_spec_kwargs,
    )
    print(f"Dataset carregado: {len(train_dataset)} amostras prontas para treino.")

    # 5. Instancia o Trainer
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=200,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=3,
        checkpoint_path=str(ckpt_dir),
        batch_size_per_gpu=args.batch_size,
        batch_size_type="sample",
        max_samples=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        logger=None,  # Treinamento limpo no terminal local
        last_per_updates=args.last_per_updates,
        log_samples=False,
        bnb_optimizer=False,
    )

    print("\nIniciando ciclo de treino via Trainer...")
    trainer.train(train_dataset, num_workers=2, resumable_with_seed=666)
    print("\nTreinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
