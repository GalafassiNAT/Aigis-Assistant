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
- **Inteligência Local e Incorporada:** Opera 100% no hardware do desenvolvedor (Windows 11, GPU AMD Radeon RX 7600), com total privacidade e sem dependência de APIs externas de terceiros.
- **Continuidade Multimodal Unificada:** Uma linha do tempo unificada em PostgreSQL (`shared_timeline`) que sincroniza a presença da assistente entre comandos de voz pelo microfone do PC e mensagens no Discord.
- **Atuadores no Sistema Operacional:** Capacidade de agir no mundo real através de um daemon persistente do PowerShell (~14ms de execução), manipulando arquivos, executando buscas e interagindo com ferramentas de produtividade (como o Obsidian Vault).
- **Voz Neural Acelerada por Hardware:** Pipeline de fala em dois estágios (Kokoro TTS + RVC v2 Voice Conversion) rodando via DirectML na GPU AMD, permitindo respostas faladas quase instantâneas.
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
- **Local & Embodied Intelligence:** Runs 100% locally on custom hardware (Windows 11, AMD Radeon RX 7600 GPU), prioritizing privacy, zero cloud subscription lock-in, and local hardware optimization.
- **Unified Multimodal Timeline:** A PostgreSQL-backed session manager (`shared_timeline`) that maintains seamless conversational continuity across PC microphone voice loops and Discord channels.
- **Real OS Actuation:** A persistent, low-latency Windows PowerShell daemon (~14ms per execution) enabling the assistant to inspect, create, edit, or delete files, launch programs, and manage an Obsidian Knowledge Vault.
- **Hardware-Accelerated Neural Voice:** Two-stage speech synthesis (Kokoro TTS + RVC v2 Voice Conversion) accelerated via DirectML on AMD RDNA3 hardware for ultra-fast time-to-first-audio.
- **Hybrid Long-Term Memory:** Relational atomic Knowledge Graph (subject-predicate-object triplets) combined with dense vector semantic retrieval (`pgvector` + FastEmbed).

### Detailed Documentation
- 📐 [System Architecture](docs/ARCHITECTURE_EN.md)
- 🚀 [Local Operation & Setup Guide](docs/OPERATION_GUIDE_EN.md)
- 🧠 [LoRA Fine-Tuning Plan](docs/LORA_FINETUNING_PLAN_EN.md)
- 🔌 [Coupling & Extensibility Analysis](docs/EXTENSIBILITY_AND_COUPLING_EN.md)

---

## 🛠️ Tech Stack & Key Technologies

- **Brain / Cognition:** [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`) with Qwen 2.5 Coder 7B (GGUF Q4_K_M).
- **Voice Pipeline:** Faster-Whisper (STT), Kokoro ONNX (TTS), RVC v2 (ONNX Runtime DirectML on AMD Radeon RX 7600).
- **Database & Storage:** PostgreSQL 16 + pgvector (via Docker).
- **Tool Automation:** Python 3.11, Persistent PowerShell Daemon, AsyncOpenAI client, Discord.py.
- **Code Quality:** Ruff (Linter & Formatter).
