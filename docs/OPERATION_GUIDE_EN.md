# Project AIGIS — Local Operation & Setup Guide

This document outlines the procedures to configure, boot, and run the **Aigis** ecosystem locally on Linux or Windows.

---

## 1. Prerequisites

- **Operating System:** Linux (Arch Linux / Ubuntu / Debian) or Windows 10/11 x64
- **Recommended Hardware:**
  - AMD Radeon GPU (e.g., RX 7600 with ROCm/DirectML/Vulkan) or NVIDIA GPU with CUDA
  - Calibrated microphone and stereo audio output
- **Software Environment:**
  - Python 3.10+ managed via Micromamba or Conda (`aigis`)
  - Docker & Docker Compose (for PostgreSQL with pgvector and Redis)
  - `llama-server` (llama.cpp compiled with GPU acceleration)

---

## 2. Service Startup Sequence

### Step 1: Launch Database & Event Bus (PostgreSQL + pgvector + Redis)
From the project root directory:
```bash
docker compose up -d
```
Two containers will start listening:
- `ai_assistant_postgres`: port **5433** (relational database, `pgvector` embeddings, shared timeline).
- `ai_assistant_redis`: port **6379** (low-latency Pub/Sub event bus and ephemeral cache).

### Step 2: Configure Google Workspace Integration (Optional but Recommended)
To enable Google Calendar, intelligent Gmail triage, and Google Drive:
1. Place your `credentials.json` file (downloaded from Google Cloud Console) in the root directory.
2. Run the local OAuth2 setup helper:
   ```bash
   python tools/setup_google_auth.py
   ```
3. The `token.json` file will be generated automatically with authorized scopes.

### Step 3: Start Local LLM Server
In a dedicated terminal, launch `llama-server` pointing to the Qwen2.5 model:
```bash
./start_brain.sh
# Or on Windows:
# .\start_brain.ps1
```

### Step 4: Launch Aigis Interfaces

- **To start the Discord Bot (Channel Hub, Proactive Daemons & Cognitive Reflection):**
  ```bash
  python interfaces/discord_bot.py
  ```
  *(The Bot automatically spawns background daemons for agenda reminders, Gmail triage, sleep/habit initiatives, and 60-second continuous timeline reflection).*
- **To run cognitive reflection manually in batch mode (optional):**
  ```bash
  python memory/worker.py
  ```
- **To start the PC Voice Assistant (Microphone + Speaker Loop):**
  ```bash
  python interfaces/voice_assistant.py
  ```
- **To start the Interactive Developer CLI:**
  ```bash
  python interfaces/cli.py
  ```

---

## 3. Environment Variables (`.env`)

The `.env` file in the root directory must include:

```ini
# PostgreSQL + pgvector
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=ai_password
POSTGRES_DB=ai_assistant
POSTGRES_PORT=5433
DATABASE_URL=postgresql://ai_user:ai_password@localhost:5433/ai_assistant

# Redis Event Bus
REDIS_URL=redis://localhost:6379/0

# Local LLM (llama-server)
LLAMA_SERVER_URL=http://localhost:8080/v1
LLAMA_MODEL_ALIAS=qwen-coder

# Discord Integration
DISCORD_BOT_TOKEN=your_token_here
DISCORD_OWNER_ID=your_discord_user_id
```

---

## 4. Code Standards & Linting

The repository enforces strict formatting via **Ruff** defined in `pyproject.toml`. Before committing changes:

```bash
# Lint check rules and import sorting
python -m ruff check core interfaces memory tools

# Format code
python -m ruff format core interfaces memory tools
```

---

## 5. Continuous Dataset Curation & Living Persona State

Aigis continuously extracts training data and evolves its emotional state through organic usage:
- **Dynamic Persona State (`data/persona_state.json`):** Tracks mood, affinity score, and learned insider vocabulary.
- **SFT Dataset (`data/training/aigis_sft.jsonl`):** Multi-turn conversational sessions automatically captured for Supervised Fine-Tuning.
- **DPO Dataset (`data/training/aigis_dpo.jsonl`):** Preference pairs (`chosen`/`rejected`) captured whenever the user corrects or critiques an assistant turn.
- For fine-tuning execution details, refer to the [LoRA Fine-Tuning Plan](LORA_FINETUNING_PLAN_EN.md).
