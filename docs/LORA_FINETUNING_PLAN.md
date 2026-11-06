# Project AIGIS — Plano de Fine-Tuning LoRA & Alinhamento de Persona

Este documento detalha o planejamento técnico para o refinamento (*fine-tuning*) do modelo neural da **Aigis** via LoRA, visando naturalidade em Português Brasileiro (PT-BR), alinhamento de persona com o criador e manutenção da precisão no uso de ferramentas (*Tool Calling*).

---

## 1. Justificativa & Diagnóstico do Problema

Atualmente, o cérebro central da Aigis utiliza o **Qwen2.5-Coder-7B-Instruct**. Embora excelente em código e execução de ferramentas, o modelo base apresenta comportamentos indesejados em linguagem natural falada:
- **Respostas Mecânicas / Robóticas:** Uso de chavões de assistente corporativo (ex: *"Entendido, combinado!"*) em situações onde o usuário apenas elogiou ou expressou alívio (*"Boa, deu certo"*).
- **Sobrecarga de Prompt:** Necessidade de dezenas de regras negativas no System Prompt (*"não peça desculpas"*, *"não recite caminhos"*), o que consome contexto e aumenta a latência de processamento de tokens.
- **Dificuldade em Resolver Referências Cruzadas:** Dificuldade em conectar pronomes (*"apaga ele"*) sem que o histórico de chamadas de ferramentas esteja perfeitamente estruturado.

---

## 2. Abordagem: LoRA (Low-Rank Adaptation)

O fine-tuning será realizado via **LoRA** congelando o modelo base e treinando adaptadores de baixo rank (*rank* $r=16$ ou $r=32$, $\alpha=32$):
- **Preservação de Capacidade:** O modelo não perde a habilidade lógica e sintática de gerar scripts PowerShell e comandos.
- **Tamanho do Artefato:** O LoRA final terá entre **30MB e 60MB**, podendo ser fundido (*merged*) diretamente no modelo base para gerar um novo arquivo `.gguf`.

---

## 3. Composição do Dataset (*Dataset Blending*) & Curadoria Contínua

O dataset de treinamento combina dados sintéticos iniciais com a **coleta autônoma e contínua** realizada em tempo real pelo curador do sistema (`memory/dataset_curator.py` e `memory/worker.py`):

| Camada | Fonte | Formato / Destino | Objetivo Principal |
| :--- | :--- | :--- | :--- |
| **1. Diálogos Reais Bem-Sucedidos (SFT)** | `memory/dataset_curator.py` | `data/training/aigis_sft.jsonl` | Diálogos reais da timeline entre Natanael e Aigis que fluíram com sucesso e naturalidade |
| **2. Pares de Preferência e Correção (DPO)** | `memory/worker.py` | `data/training/aigis_dpo.jsonl` | Pares `{prompt, chosen, rejected}` gerados sempre que Natanael corrige um erro ou tom da Aigis |
| **3. Fluência Conversacional PT-BR** | `Canarim-Instruct-PTBR` / Hugging Face | Externo / Base | Suporte de vocabulário e prosódia informal brasileira |
| **4. Persona & Tool Calling Especializado** | Sintético (`training/generate_aigis_data.py`) | `data/training/synthetic_aigis.jsonl` | Identidade fundacional, ferramentas e feedback de voz |

---

## 4. Pipeline de Treinamento Contínuo

```mermaid
flowchart LR
    TL[shared_timeline / Interações Reais] --> CW[Cognitive Worker & Curator]
    CW --> SFT[data/training/aigis_sft.jsonl]
    CW --> DPO[data/training/aigis_dpo.jsonl]
    SFT --> Blend[Compilação do Dataset]
    DPO --> Blend
    Blend --> Colab[Google Colab: Unsloth / GPU T4/A100]
    Colab --> LoRA[Adaptador LoRA / DPO Treinado]
    LoRA --> Merge[Merge GGUF: aigis-7b-q4_k_m.gguf]
    Merge --> LlamaServer[Deploy no llama-server Local]
```

1. **Compilação:** Unificação dos dados em formato ChatML compatível com Qwen:
   ```json
   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```
2. **Treinamento no Colab:** Execução com `unsloth` em GPU gratuita T4/A100 (tempo estimado: 15–20 minutos).
3. **Quantização e Deploy:** Exportação em formato GGUF `Q4_K_M` e substituição no `llama-server`.
