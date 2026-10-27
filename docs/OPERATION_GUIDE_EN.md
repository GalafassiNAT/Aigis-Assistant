# Project AIGIS — Local Operation & Setup Guide

This document outlines the procedures to configure, boot, and run the **Aigis** ecosystem locally on Windows.

---

## 1. Prerequisites

- **Operating System:** Windows 10/11 x64
- **Recommended Hardware:**
  - AMD Radeon GPU (e.g., RX 7600) or NVIDIA GPU with DirectML / ROCm / CUDA support
  - Calibrated microphone and stereo audio output
- **Software Environment:**
  - Python 3.11 managed via Conda or Micromamba (`ai-assistant`)
  - Docker Desktop (for PostgreSQL with pgvector)
  - `llama-server.exe` (llama.cpp compiled with ROCm/DirectML/Vulkan)

---

## 2. Service Startup Sequence

### Step 1: Launch Database (PostgreSQL + pgvector)
From the project root directory:
```powershell
docker compose up -d
```
The `ai_assistant_postgres` container will start listening on port **5433**.

### Step 2: Start Local LLM Server
In a dedicated terminal, launch `llama-server` pointing to the Qwen2.5-Coder model:
```powershell
llama-server.exe -m "path\to\qwen2.5-coder-7b-instruct-q4_k_m.gguf" --port 8080 -ngl 99 --alias qwen-coder -c 8192
```

### Step 3: Launch Aigis Interfaces

- **To start the PC Voice Assistant (Microphone + Speaker Loop):**
  ```powershell
  python interfaces/voice_assistant.py
  ```
- **To start the Discord Bot:**
  ```powershell
  python interfaces/discord_bot.py
  ```
- **To start the Interactive Developer CLI:**
  ```powershell
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

```powershell
# Lint check rules and import sorting
python -m ruff check core interfaces memory tools

# Format code
python -m ruff format core interfaces memory tools
```
