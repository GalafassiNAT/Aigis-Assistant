# Project AIGIS — LoRA Fine-Tuning & Persona Alignment Plan

This document outlines the technical strategy for fine-tuning **Aigis's** neural model using LoRA. The objective is to achieve natural conversational flow in Brazilian Portuguese (PT-BR), deep alignment with the creator, and robust tool-calling accuracy.

---

## 1. Problem Statement & Motivation

Currently, Aigis's core cognition runs on **Qwen2.5-Coder-7B-Instruct**. While state-of-the-art for coding and function execution, the base model exhibits conversational deficiencies in spoken dialogue:
- **Mechanical Responses:** Over-reliance on generic corporate assistant clichés (e.g., *"Understood, agreed!"*) when the user is expressing praise or relief (*"Nice, it worked"*).
- **Prompt Bloat:** Excessive negative instructions in the system prompt (*"do not apologize"*, *"do not recite paths"*), inflating prompt evaluation latency and token costs.
- **Reference Resolution:** Difficulty resolving contextual pronouns (*"delete it"*) unless complete structured tool call histories are persistently maintained.

---

## 2. Approach: LoRA (Low-Rank Adaptation)

Fine-tuning will be performed using **LoRA** on the frozen base model, adjusting low-rank adapter matrices ($r=16$ or $r=32$, $\alpha=32$):
- **Capability Preservation:** Prevents catastrophic forgetting of coding logic and PowerShell generation.
- **Artifact Footprint:** The resulting LoRA weighs **30MB–60MB**, which can be dynamically mounted or fused directly into a single `.gguf` binary.

---

## 3. Dataset Composition (*Dataset Blending*)

The training corpus will combine three distinct layers:

| Layer | Source | Estimated Count | Primary Goal |
| :--- | :--- | :--- | :--- |
| **1. PT-BR Conversational Fluency** | `Canarim-Instruct-PTBR` / `Pt-Corpus-Instruct` (Hugging Face) | ~500–800 turns | Natural Brazilian Portuguese phrasing and short dialogue turns |
| **2. Social Dynamics & Roleplay** | `agentlans/combined-roleplay` / Curated samples | ~200–300 turns | Proportional responses to compliments, feedback, and relief |
| **3. Aigis Persona & Tool Calling** | Domain Synthetic Generator (`training/generate_aigis_data.py`) | ~250–400 turns | Creator identity with Natanael, Windows PowerShell tool use, concise voice cues |

---

## 4. Execution Pipeline

```mermaid
flowchart LR
    HF[Hugging Face Datasets] --> Filter[PT-BR Filter Script]
    Syn[Aigis Synthetic Generator] --> Blend[Merge into aigis_dataset.jsonl]
    Filter --> Blend
    Blend --> Colab[Google Colab: Unsloth / T4 GPU]
    Colab --> LoRA[Trained LoRA Weights]
    LoRA --> Merge[Merge GGUF: aigis-7b-q4_k_m.gguf]
    Merge --> LlamaServer[Local llama-server Deployment]
```

1. **Compilation:** Standardize into ChatML structure:
   ```json
   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```
2. **Training in Colab:** Utilize `unsloth` on free T4/A100 GPU instances (~15–20 min training time).
3. **Quantization & Deployment:** Export GGUF `Q4_K_M` binary directly into local model storage.
