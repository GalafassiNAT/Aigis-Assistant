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
- **Continuidade Multimodal Unificada:** Uma linha do tempo unificada em PostgreSQL (`shared_timeline`) e barramento Redis Pub/Sub (`core/bus.py`) que sincroniza a presença da assistente em tempo real (<1ms) entre comandos de voz no PC e canais do Discord.
- **Hub Proativo & Gestão de Vida (Google Workspace):**
  - **Google Calendar:** Agendamento inteligente com resolução de linguagem natural (dias da semana), eventos recorrentes (`RRULE:FREQ=WEEKLY;BYDAY=...`), coloração temática dinâmica (`colorId`) e exclusão idempotente. Lembretes automáticos com pings no Discord (`#aigis-agenda`) 15 a 30 minutos antes de compromissos.
  - **Triagem Inteligente de Gmail:** Filtragem profunda de newsletters e anúncios via cabeçalhos RFC (`List-Unsubscribe`, `List-Id`) e detecção heurística de remetentes automatizados. Alertas visuais ricos no Discord (`#aigis-inbox`) com menção ativa para e-mails urgentes e deduplicação persistente em disco (`data/notified_emails.json`).
  - **Leitura Verificada de E-mails:** Ferramenta dedicada (`read_email_body`) com guardrail anti-alucinação no LLM, forçando a leitura factual e inline do conteúdo real sem invenções.
  - **Google Drive:** Busca semântica e leitura de documentos no Workspace.
- **Atuadores no Sistema Operacional & Tool Calling Verificado:** Capacidade de agir no mundo real via ferramentas nativas verificadas no disco (`create_file`, `delete_file`, `list_files`, `read_file`) e execução de comandos shell com timeout resiliente e guardrail anti-alucinação.
- **Voz Neural em Tempo Real Acelerada por GPU:** Pipeline de fala residente em VRAM (Piper Dii / Kokoro ONNX + Trato Vocal RVC v2 40kHz em PyTorch GPU), com feedback imediato (<50ms), streaming por sentença e normalizador fonético de termos técnicos (TypeScript, Python, Docker, etc.).
- **Memória Permanente Híbrida & Autobiográfica:** Grafo de Conhecimento relacional (tripletos atômicos) combinado com busca vetorial densa (`pgvector` + FastEmbed), histórico episódico (`episodic_memories`) e perfil sociopsicológico e hábitos do usuário (`user_profiles`).
- **Persona Viva & Parceria Cognitiva:**
  - **Estado Dinâmico da Persona (`core/persona.py`):** Monitora humor, nível de afinidade e termos compartilhados (`data/persona_state.json`), evoluindo organicamente a cada interação.
  - **Modo Co-piloto de Ideias:** Injeção de contexto cognitivo dinâmico preservando o KV cache do `llama-server`, tornando a Aigis uma parceira ativa para discussões técnicas, projetos e decisões de vida.
  - **Iniciativa Autônoma Proativa:** Ações espontâneas sem comando prévio, como vigilância de sono e bem-estar em madrugadas de trabalho intenso e checagem de hábitos diários (Mandarim, LeetCode) com cooldown anti-spam no Redis.
  - **Curadoria Contínua de Datasets:** Mineração autônoma de conversas pelo worker cognitivo (`memory/worker.py`), salvando diálogos de alta qualidade para SFT (`data/training/aigis_sft.jsonl`) e pares de preferência de correções do usuário para DPO (`data/training/aigis_dpo.jsonl`).

### Documentação Detalhada
- 📐 [Arquitetura do Sistema](docs/ARCHITECTURE.md)
- 🎓 [Proposta de Pesquisa / TCC (UTFPR)](docs/TCC_RESEARCH_PROPOSAL.md)
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
- **Unified Multimodal Timeline:** A PostgreSQL-backed session manager (`shared_timeline`) paired with a real-time Redis Pub/Sub Event Bus (`core/bus.py`) that maintains instantaneous (<1ms) conversational continuity across PC microphone voice loops and Discord channels.
- **Proactive Life Hub & Google Workspace Hub:**
  - **Google Calendar:** Natural language date parsing (weekdays), recurrent event scheduling (`RRULE:FREQ=WEEKLY;BYDAY=...`), thematic category color coding (`colorId`), and conflict-safe deletion. Proactive alerts with owner mentions in `#aigis-agenda` 15 to 30 minutes before events.
  - **Intelligent Gmail Triage:** Strict RFC header filtering (`List-Unsubscribe`, `List-Id`) and heuristic automated sender blocks. Rich visual Discord embeds (`#aigis-inbox`) with active pings for high-priority emails and persistent disk deduplication (`data/notified_emails.json`).
  - **Verified Email Reading:** Dedicated tool (`read_email_body`) backed by strict inline tool-calling guardrails, eliminating hallucinations and ensuring verbatim email extraction.
  - **Google Drive:** Semantic search and document reading within user workspace notes.
- **Real OS Actuation & Verified Tool Calling:** Operating system actuation via verified filesystem tools (`create_file`, `delete_file`, `list_files`, `read_file`), strict timeout enforcement on persistent shell execution, and anti-hallucination guardrails.
- **Hardware-Accelerated Neural Voice:** In-VRAM neural speech pipeline (Piper Dii / Kokoro ONNX + In-VRAM RVC v2 40kHz PyTorch GPU vocal tract) with sub-50ms acoustic feedback, sentence streaming, and real-time technical Anglicism phonetic normalization.
- **Hybrid & Autobiographical Long-Term Memory:** Relational atomic Knowledge Graph (triplets) combined with dense vector semantic retrieval (`pgvector` + FastEmbed), episodic interaction logs (`episodic_memories`), and sociological/psychological user profiling (`user_profiles`).
- **Living Persona & Cognitive Co-Pilot:**
  - **Dynamic Living Persona (`core/persona.py`):** Tracks mood, affinity score, shared idioms, and behavioral adaptations (`data/persona_state.json`), evolving organically with interaction history.
  - **Co-Pilot Brainstorming Mode:** Dynamic cognitive context injection at prompt tail preserving `llama-server` prefix KV cache, allowing deep technical, architectural, and life partnership dialogues.
  - **Autonomous Initiative Engine:** Spontaneous outreach without user triggers (late-night sleep/health guard, post-habit check-ins for Mandarin and LeetCode) protected by Redis cooldown locks.
  - **Continuous Dataset Curation:** Autonomous mining by the background reflection worker (`memory/worker.py`), recording high-quality dialogues for SFT (`data/training/aigis_sft.jsonl`) and user correction pairs for DPO (`data/training/aigis_dpo.jsonl`).

### Detailed Documentation
- 📐 [System Architecture](docs/ARCHITECTURE_EN.md)
- 🎓 [Capstone & Research Proposal (UTFPR)](docs/TCC_RESEARCH_PROPOSAL_EN.md)
- 🚀 [Local Operation & Setup Guide](docs/OPERATION_GUIDE_EN.md)
- 🧠 [LoRA Fine-Tuning Plan](docs/LORA_FINETUNING_PLAN_EN.md)
- 🔌 [Coupling & Extensibility Analysis](docs/EXTENSIBILITY_AND_COUPLING_EN.md)

---

## 🛠️ Tech Stack & Key Technologies

- **Brain / Cognition:** [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`) with Qwen 2.5 7B Instruct (GGUF Q4_K_M, 8192-token context).
- **Voice Pipeline:** Whisper (STT with VAD filtering), Piper Dii (PT-BR) & Kokoro ONNX (EN), In-VRAM RVC v2 40kHz Vocal Tract (PyTorch GPU / ROCm / Vulkan), Realtime Anglicism Normalizer.
- **Database, State & Event Bus:** PostgreSQL 16 + pgvector (port 5433) and Redis 7 (port 6379) via Docker.
- **Cognitive Engine & Training:** Autobiographical Memory (`user_profiles`, `episodic_memories`), Dynamic Living Persona (`data/persona_state.json`), Continuous SFT/DPO Dataset Curation (`data/training/`).
- **Workspace & Life Hub:** Google Calendar API v3, Gmail API v1, Google Drive API v3 (OAuth 2.0 with local token rotation), Obsidian Local Vault integration.
- **Tool Automation:** Python 3.10+, Verified File Operators, Persistent PowerShell Daemon (with threaded timeout deadline), Bash Subprocess Runner, Discord.py.
- **Code Quality:** Ruff (Linter & Formatter).
