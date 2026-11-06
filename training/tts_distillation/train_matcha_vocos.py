"""
Aigis Sweetspot Voice Distillation: Matcha-TTS (100 mels) + Vocos 24kHz.
Ultra-safe local training script with strict desktop freeze prevention.
"""

import os
import sys
import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
from omegaconf import OmegaConf

# 1. System safeguards for zero desktop freezing
try:
    os.nice(15)  # Set low CPU scheduling priority
    print("[*] Set CPU scheduling priority: nice 15")
except Exception as e:
    print(f"[-] Could not set nice level: {e}")

if torch.cuda.is_available():
    # Hard cap VRAM allocation to 35% of total (leaves > 5 GB free for display server)
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    print(f"[*] Set GPU VRAM allocation cap: 35% (approx 2.8 GB max)")

from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.model import fix_len_compatibility
import matcha.text as mt
from matcha.utils.utils import intersperse
from vocos import Vocos

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "training/tts_distillation/dataset/matcha_vocos_dataset.pt"
BASE_CKPT_PATH = BASE_DIR / "models/matcha/matcha_ljspeech.ckpt"
CHECKPOINT_DIR = BASE_DIR / "models/matcha"
OUTPUT_AUDIO_DIR = BASE_DIR / "output/audio/test_rates"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

class InRamDataset(Dataset):
    def __init__(self, items, mel_mean, mel_std):
        self.items = items
        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        norm_mel = (item["mel"] - self.mel_mean) / self.mel_std
        return {
            "id": item["id"],
            "x": item["x"],
            "mel": norm_mel,
            "text": item["text"],
        }

def collate_fn(batch):
    B = len(batch)
    x_lens = [item["x"].shape[0] for item in batch]
    y_lens = [item["mel"].shape[1] for item in batch]

    max_x = max(x_lens)
    max_y = fix_len_compatibility(max(y_lens))
    n_feats = batch[0]["mel"].shape[0]

    x_padded = torch.zeros((B, max_x), dtype=torch.long)
    y_padded = torch.zeros((B, n_feats, max_y), dtype=torch.float32)

    for i, item in enumerate(batch):
        x_padded[i, :x_lens[i]] = item["x"]
        y_padded[i, :, :y_lens[i]] = item["mel"]

    x_lengths = torch.tensor(x_lens, dtype=torch.long)
    y_lengths = torch.tensor(y_lens, dtype=torch.long)

    return {
        "x": x_padded,
        "x_lengths": x_lengths,
        "y": y_padded,
        "y_lengths": y_lengths,
        "ids": [item["id"] for item in batch],
    }

def create_model(n_feats=100, mel_mean=-1.9395, mel_std=3.2207):
    cfg = OmegaConf.create({
        'n_vocab': 178,
        'n_spks': 1,
        'spk_emb_dim': 64,
        'n_feats': n_feats,
        'encoder': {
            'encoder_type': 'RoPE Encoder',
            'encoder_params': {
                'n_feats': n_feats,
                'n_channels': 192,
                'filter_channels': 768,
                'filter_channels_dp': 256,
                'n_heads': 2,
                'n_layers': 6,
                'kernel_size': 3,
                'p_dropout': 0.1,
                'spk_emb_dim': 64,
                'n_spks': 1,
                'prenet': True
            },
            'duration_predictor_params': {
                'filter_channels_dp': 256,
                'kernel_size': 3,
                'p_dropout': 0.1
            }
        },
        'decoder': {
            'channels': [256, 256],
            'dropout': 0.05,
            'attention_head_dim': 64,
            'n_blocks': 1,
            'num_mid_blocks': 2,
            'num_heads': 2,
            'act_fn': 'snakebeta'
        },
        'cfm': {
            'name': 'CFM',
            'solver': 'euler',
            'sigma_min': 0.0001
        },
        'data_statistics': {'mel_mean': mel_mean, 'mel_std': mel_std},
        'out_size': None
    })
    return MatchaTTS(**cfg)

def warm_start_weights(model, ckpt_path):
    print(f"[*] Warm-starting from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd_src = ckpt['state_dict']
    sd_tgt = model.state_dict()

    transferred = 0
    adapted = 0
    for k, v in sd_tgt.items():
        if k in sd_src:
            if v.shape == sd_src[k].shape:
                sd_tgt[k].copy_(sd_src[k])
                transferred += 1
            else:
                src = sd_src[k]
                if k == 'encoder.proj_m.weight':
                    sd_tgt[k][:80].copy_(src)
                elif k == 'encoder.proj_m.bias':
                    sd_tgt[k][:80].copy_(src)
                elif k == 'decoder.estimator.final_proj.weight':
                    sd_tgt[k][:80].copy_(src)
                elif k == 'decoder.estimator.final_proj.bias':
                    sd_tgt[k][:80].copy_(src)
                elif k == 'decoder.estimator.down_blocks.0.0.res_conv.weight':
                    sd_tgt[k][:, :80].copy_(src[:, :80])
                    sd_tgt[k][:, 100:180].copy_(src[:, 80:160])
                elif k == 'decoder.estimator.down_blocks.0.0.block1.block.0.weight':
                    sd_tgt[k][:, :80].copy_(src[:, :80])
                    sd_tgt[k][:, 100:180].copy_(src[:, 80:160])
                elif k == 'decoder.estimator.time_mlp.linear_1.weight':
                    sd_tgt[k][:, :160].copy_(src)
                adapted += 1

    model.load_state_dict(sd_tgt)
    print(f"[+] Loaded {transferred} exact layers and adapted {adapted} layers successfully.")

def synthesize_test(model, vocos, device, prompt_ipa, save_path, mel_mean, mel_std):
    model.eval()
    with torch.inference_mode():
        seq, _ = mt.text_to_sequence(prompt_ipa, [])
        seq_interspersed = intersperse(seq, 0)
        x = torch.IntTensor(seq_interspersed).unsqueeze(0).to(device)
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)

        out = model.synthesise(x, x_lengths, n_timesteps=4, length_scale=0.95)
        mel = out["mel"]  # denormalized
        wav = vocos.decode(mel).squeeze().cpu().numpy()
        sf.write(str(save_path), wav, 24000)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    # Load dataset
    print(f"[*] Loading dataset from {DATASET_PATH}...")
    dataset_dict = torch.load(DATASET_PATH, map_location="cpu")
    stats = dataset_dict["stats"]
    mel_mean = stats["mel_mean"]
    mel_std = stats["mel_std"]
    n_feats = stats["n_feats"]

    train_set = InRamDataset(dataset_dict["train"], mel_mean, mel_std)
    val_set = InRamDataset(dataset_dict["val"], mel_mean, mel_std)
    print(f"[+] Train dataset: {len(train_set)} samples | Val dataset: {len(val_set)} samples")

    # Safe batch size: 8
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Initialize model & warm start
    model = create_model(n_feats=n_feats, mel_mean=mel_mean, mel_std=mel_std).to(device)
    warm_start_weights(model, BASE_CKPT_PATH)

    # Load Vocos for test audio generation
    print("[*] Loading Vocos 24kHz for test evaluations...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    # Benchmark test sentence IPA (clean pt-br)
    test_sentence_ipa = "olˈa, mˈɛstry! kˌomʊ pˌɔsʊ ˌaʒudˈar vosˌe ˈoʒy?"

    # Training parameters
    total_epochs = 250
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    start_total_time = time.time()

    print(f"\n{'='*70}")
    print(f" Starting Aigis Sweetspot Matcha-TTS Training (Total Epochs: {total_epochs})")
    print(f"{'='*70}\n")

    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss_total = 0.0
        dur_loss_total = 0.0
        prior_loss_total = 0.0
        diff_loss_total = 0.0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            x_lengths = batch["x_lengths"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            y_lengths = batch["y_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            dur_loss, prior_loss, diff_loss, _ = model(x, x_lengths, y, y_lengths)
            loss = dur_loss + prior_loss + diff_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            dur_loss_total += dur_loss.item()
            prior_loss_total += prior_loss.item()
            diff_loss_total += diff_loss.item()

        scheduler.step()

        n_batches = len(train_loader)
        avg_train_loss = train_loss_total / n_batches
        avg_dur = dur_loss_total / n_batches
        avg_prior = prior_loss_total / n_batches
        avg_diff = diff_loss_total / n_batches
        epoch_time = time.time() - epoch_start_time

        # Evaluate validation loss every 25 epochs
        if epoch % 25 == 0 or epoch == total_epochs:
            model.eval()
            val_loss_total = 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    x_lengths = batch["x_lengths"].to(device)
                    y = batch["y"].to(device)
                    y_lengths = batch["y_lengths"].to(device)

                    v_dur, v_prior, v_diff, _ = model(x, x_lengths, y, y_lengths)
                    val_loss_total += (v_dur + v_prior + v_diff).item()

            avg_val_loss = val_loss_total / len(val_loader)
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0

            print(
                f"[Epoch {epoch:03d}/{total_epochs}] "
                f"Train: {avg_train_loss:.4f} (dur: {avg_dur:.3f}, pri: {avg_prior:.3f}, dif: {avg_diff:.3f}) | "
                f"Val: {avg_val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {epoch_time:.2f}s | "
                f"VRAM: {vram_mb:.1f} MB"
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = CHECKPOINT_DIR / "aigis_matcha_vocos_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "stats": stats,
                }, best_path)

            # Synthesize milestone audio every 50 epochs
            if epoch % 50 == 0 or epoch == total_epochs:
                test_audio_path = OUTPUT_AUDIO_DIR / f"aigis_matcha_epoch_{epoch:03d}.wav"
                synthesize_test(model, vocos, device, test_sentence_ipa, test_audio_path, mel_mean, mel_std)
                print(f"  --> [Audio Milestone] Rendered test sample: {test_audio_path}")
        elif epoch % 5 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:03d}/{total_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} (dur: {avg_dur:.3f}, pri: {avg_prior:.3f}, dif: {avg_diff:.3f}) | "
                f"Time: {epoch_time:.2f}s"
            )

    # Save last model
    last_path = CHECKPOINT_DIR / "aigis_matcha_vocos_last.pt"
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "stats": stats,
    }, last_path)

    total_time = time.time() - start_total_time
    print(f"\n{'='*70}")
    print(f" Training Complete in {total_time/60:.2f} minutes!")
    print(f" Best Validation Loss: {best_val_loss:.4f}")
    print(f" Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
