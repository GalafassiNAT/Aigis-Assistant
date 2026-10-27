"""
Script de unificação e mesclagem final dos dados de treinamento do LoRA.
Gera o dataset final: training/data/aigis_finetune_chatml.jsonl
"""

import json
from pathlib import Path
import random

DATA_DIR = Path("training/data")
FINAL_OUTPUT = DATA_DIR / "aigis_finetune_chatml.jsonl"


def merge_datasets():
    all_examples = []

    # 1. Carrega dados de Persona e Ferramentas da Aigis (PT-BR)
    aigis_file = DATA_DIR / "aigis_persona_samples.jsonl"
    if aigis_file.exists():
        with open(aigis_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))
        print(f"[Merge] Aigis Persona: {len(all_examples)} amostras carregadas.")

    # 2. Carrega amostras do Hermes Function Calling (Inglês)
    hermes_file = DATA_DIR / "hermes_samples.jsonl"
    hermes_count = 0
    if hermes_file.exists():
        with open(hermes_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))
                    hermes_count += 1
        print(f"[Merge] Hermes Tool Calling: {hermes_count} amostras carregadas.")

    # Embaralha os dados para garantir convergência equilibrada
    random.seed(42)
    random.shuffle(all_examples)

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[Sucesso] Dataset final compilado com {len(all_examples)} exemplos!")
    print(f"Arquivo gerado: {FINAL_OUTPUT.resolve()}")


if __name__ == "__main__":
    merge_datasets()
