"""
Pipeline de extração e filtragem de dados públicos para o fine-tuning da Aigis.
Baixa amostras selecionadas do Hugging Face via streaming/HTTP para evitar downloads pesados.
"""

import json
from pathlib import Path
import httpx

OUTPUT_DIR = Path("training/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_hermes_function_calling_samples(limit: int = 350) -> list[dict]:
    """
    Baixa uma fatia de exemplos de alta qualidade de Function Calling em Inglês
    do NousResearch/hermes-function-calling-v1 (glaive-function-calling-5k.json).
    """
    print(f"[Download] Baixando amostras do Hermes Function Calling (limite: {limit})...")
    url = "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main/glaive-function-calling-5k.json"

    samples = []
    try:
        r = httpx.get(url, follow_redirects=True, timeout=60.0)
        if r.status_code == 200:
            data = r.json()
            for item in data[:limit]:
                conversations = item.get("conversations", [])
                if conversations and len(conversations) >= 2:
                    msgs = []
                    for msg in conversations:
                        role = "user" if msg.get("from") == "human" else "assistant"
                        msgs.append({"role": role, "content": msg.get("value", "")})
                    samples.append({"messages": msgs})
        print(f"[OK] Hermes: {len(samples)} exemplos de tool calling coletados.")
    except Exception as e:
        print(f"[Aviso] Erro ao baixar Hermes: {e}")

    return samples


def download_ptbr_conversational_samples(limit: int = 400) -> list[dict]:
    """
    Coleta amostras conversacionais naturais em Português Brasileiro (PT-BR).
    Foca em trocas curtas e naturais.
    """
    print(f"📥 Baixando amostras de diálogos PT-BR (limite: {limit})...")
    url = "https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct/resolve/main/data/train-00000-of-00001.parquet"
    # Fallback seguro com amostras sintéticas de conversação brasileira se parquet for lento
    samples = []
    # Usaremos os dados estruturados de instrução multi-turn
    return samples


if __name__ == "__main__":
    hermes = download_hermes_function_calling_samples(limit=300)
    out_file = OUTPUT_DIR / "hermes_samples.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for item in hermes:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Salvo em: {out_file}")
