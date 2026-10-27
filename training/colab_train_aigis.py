# -*- coding: utf-8 -*-
"""
Google Colab Notebook (Unsloth) para Fine-Tuning LoRA da Aigis
Modelo Base: Qwen/Qwen2.5-Coder-7B-Instruct
Hardware: GPU T4 gratuita ou A100

Como usar:
1. Abra https://colab.research.google.com
2. Conecte em um ambiente com GPU (Ambiente de Execução > Alterar tipo > T4 GPU)
3. Cole e execute as células abaixo.
4. Faça o upload do arquivo `aigis_finetune_chatml.jsonl` gerado pelo projeto.
"""

# ==========================================
# CÉLULA 1: Instalação do Unsloth & Dependências
# ==========================================
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps trl peft accelerate bitsandbytes

# ==========================================
# CÉLULA 2: Carregamento do Modelo Base Qwen
# ==========================================
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # Auto detecção (Float16 ou Bfloat16)
load_in_4bit = True  # 4bit QLoRA para caber perfeitamente na T4 (consome apenas ~5GB VRAM)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ==========================================
# CÉLULA 3: Configuração do Adaptador LoRA
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank do LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,  # 0 otimizado para Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# ==========================================
# CÉLULA 4: Carregar o Dataset da Aigis
# ==========================================
from datasets import load_dataset

dataset = load_dataset("json", data_files="aigis_finetune_chatml.jsonl", split="train")

def format_prompts(batch):
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in batch["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# ==========================================
# CÉLULA 5: Treinamento do LoRA (~10 a 15 min)
# ==========================================
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=120,  # Suficiente para 320 amostras convergirem
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

# ==========================================
# CÉLULA 6: Salvar e Exportar GGUF (Q4_K_M)
# ==========================================
# Salva diretamente no formato pronto para o llama-server da Aigis no Windows:
model.save_pretrained_gguf("aigis-qwen-7b-lora", tokenizer, quantization_method="q4_k_m")
print("Modelo Aigis exportado com sucesso em GGUF!")
