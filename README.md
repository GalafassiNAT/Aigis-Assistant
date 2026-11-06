# Aigis Assistant 🤖✨

<p align="center">
  <a href="#-sobre-o-projeto-pt-br">Português</a> •
  <a href="#-about-the-project-en">English</a>
</p>

---

## 🇧🇷 Sobre o Projeto (PT-BR)

> **Nota de Contexto e Finalidade:**
> Este é um **projeto experimental e de aprendizado contínuo**. O objetivo principal é a exploração prática e o entendimento aprofundado de **engenharia de sistemas autônomos com LLMs locais**, estruturas cognitivas, ferramentas operacionais (*tool use / function calling*), integração multimodal (voz de baixíssima latência) e técnicas de alinhamento e treinamento (como **LoRA / Fine-Tuning**).

### A Ideia por trás da Aigis
A maioria dos assistentes de inteligência artificial atuais reside na nuvem e opera como chatbots passivos e isolados do ambiente do usuário. O projeto **Aigis** foi idealizado para explorar um paradigma diferente:
- **Inteligência Local e Incorporada:** Opera 100% no hardware do desenvolvedor (Linux ou Windows 11, GPU AMD Radeon RX 7600), com total privacidade e sem dependência de APIs externas de terceiros.
- **Continuidade Multimodal Unificada:** Uma linha do tempo unificada em PostgreSQL (`shared_timeline`) que sincroniza a presença da assistente entre comandos de voz pelo microfone do PC e mensagens no Discord.
- **Atuadores no Sistema Operacional & Tool Calling Verificado:** Capacidade de agir no mundo real via ferramentas nativas verificadas no disco (`create_file`, `delete_file`, `list_files`, `read_file`) e execução de comandos com timeout resiliente, guardrail anti-alucinação e recuperação automática de sessões persistentes.
- **Voz Neural em Tempo Real Acelerada por GPU:** Pipeline de fala residente em VRAM (Piper Dii / Kokoro ONNX + Trato Vocal RVC v2 40kHz em PyTorch GPU), com feedback imediato (<50ms), streaming por sentença e normalizador fonético de termos técnicos (TypeScript, Python, Docker, etc.).
- **Memória Permanente Híbrida:** Grafo de Conhecimento relacional (tripletos atômicos) combinado com busca vetorial densa (`pgvector` + FastEmbed).

### Documentação Detalhada
- 📐 [Arquitetura do Sistema](docs/ARCHITECTURE.md)
- 🚀 [Guia de Operação e Setup Local](docs/OPERATION_GUIDE.md)
- 🧠 [Plano de Fine-Tuning LoRA](docs/LORA_FINETUNING_PLAN.md)
- 🔌 [Análise de Acoplamento e Extensibilidade](docs/EXTENSIBILITY_AND_COUPLING.md)

---

## 🇺🇸 About the Project (EN)

> **Context & Educational Purpose:**
> This is an **experimental, research, and learning project**. The core objective is hands-on exploration and deep understanding of **local autonomous AI systems**, agentic cognitive architectures, tool use / function calling, low-latency multimodal voice pipelines, and neural model alignment/training techniques (such as **LoRA fine-tuning**).

### The Vision behind Aigis
Most modern AI assistants live in the cloud as passive, isolated text chatbots. The **Aigis** project explores a fundamentally different approach:
- **Local & Embodied Intelligence:** Runs 100% locally on custom hardware (Linux / Windows 11, AMD Radeon RX 7600 GPU), prioritizing privacy, zero cloud subscription lock-in, and local hardware optimization.
- **Unified Multimodal Timeline:** A PostgreSQL-backed session manager (`shared_timeline`) that maintains seamless conversational continuity across PC microphone voice loops and Discord channels.
- **Real OS Actuation & Verified Tool Calling:** Operating system actuation via verified filesystem tools (`create_file`, `delete_file`, `list_files`, `read_file`), strict timeout enforcement on persistent shell execution, and anti-hallucination guardrails.
- **Hardware-Accelerated Neural Voice:** In-VRAM neural speech pipeline (Piper Dii / Kokoro ONNX + In-VRAM RVC v2 40kHz PyTorch GPU vocal tract) with sub-50ms acoustic feedback, sentence streaming, and real-time technical Anglicism phonetic normalization.
- **Hybrid Long-Term Memory:** Relational atomic Knowledge Graph (subject-predicate-object triplets) combined with dense vector semantic retrieval (`pgvector` + FastEmbed).

### Detailed Documentation
- 📐 [System Architecture](docs/ARCHITECTURE_EN.md)
- 🚀 [Local Operation & Setup Guide](docs/OPERATION_GUIDE_EN.md)
- 🧠 [LoRA Fine-Tuning Plan](docs/LORA_FINETUNING_PLAN_EN.md)
- 🔌 [Coupling & Extensibility Analysis](docs/EXTENSIBILITY_AND_COUPLING_EN.md)

---

## 🛠️ Tech Stack & Key Technologies

- **Brain / Cognition:** [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`) with Qwen 2.5 7B Instruct (GGUF Q4_K_M, 8192-token context).
- **Voice Pipeline:** Whisper (STT with VAD filtering), Piper Dii (PT-BR) & Kokoro ONNX (EN), In-VRAM RVC v2 40kHz Vocal Tract (PyTorch GPU / ROCm / Vulkan), Realtime Anglicism Normalizer.
- **Database & Storage:** PostgreSQL 16 + pgvector (via Docker).
- **Tool Automation:** Python 3.10+, Verified File Operators, Persistent PowerShell Daemon (with threaded timeout deadline), Bash Subprocess Runner, Discord.py.
- **Code Quality:** Ruff (Linter & Formatter).
