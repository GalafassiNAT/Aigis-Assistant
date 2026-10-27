# Project AIGIS — Guia de Operação & Setup Local

Este documento descreve como configurar, iniciar e operar o ecossistema completo da **Aigis** localmente no Windows.

---

## 1. Pré-Requisitos

- **Sistema Operacional:** Windows 10/11 x64
- **Hardware Recomendado:**
  - GPU AMD Radeon (ex: RX 7600) ou NVIDIA com suporte DirectML / ROCm / CUDA
  - Microfone calibrado e saída de áudio estéreo
- **Ambiente de Software:**
  - Python 3.11 gerenciado via Conda ou Micromamba (`ai-assistant`)
  - Docker Desktop (para o PostgreSQL com pgvector)
  - `llama-server.exe` (llama.cpp compilado com ROCm/DirectML/Vulkan)

---

## 2. Inicialização dos Serviços

### Passo 1: Subir o Banco de Dados (PostgreSQL + pgvector)
No diretório raiz do projeto:
```powershell
docker compose up -d
```
O container `ai_assistant_postgres` será iniciado na porta **5433**.

### Passo 2: Iniciar o Servidor LLM Local
Em um terminal dedicado, inicie o `llama-server` apontando para o modelo Qwen2.5-Coder:
```powershell
llama-server.exe -m "caminho\para\qwen2.5-coder-7b-instruct-q4_k_m.gguf" --port 8080 -ngl 99 --alias qwen-coder -c 8192
```

### Passo 3: Iniciar as Interfaces da Aigis

- **Para usar o Assistente de Voz no PC (Microfone + Alto-falante):**
  ```powershell
  python interfaces/voice_assistant.py
  ```
- **Para iniciar o Bot do Discord:**
  ```powershell
  python interfaces/discord_bot.py
  ```
- **Para o terminal CLI interativo:**
  ```powershell
  python interfaces/cli.py
  ```

---

## 3. Variáveis de Ambiente (`.env`)

O arquivo `.env` na raiz deve conter as configurações principais:

```ini
# PostgreSQL + pgvector
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=ai_password
POSTGRES_DB=ai_assistant
POSTGRES_PORT=5433
DATABASE_URL=postgresql://ai_user:ai_password@localhost:5433/ai_assistant

# LLM Local (llama-server)
LLAMA_SERVER_URL=http://localhost:8080/v1
LLAMA_MODEL_ALIAS=qwen-coder

# Integração Discord
DISCORD_BOT_TOKEN=seu_token_aqui
DISCORD_OWNER_ID=seu_id_de_usuario
```

---

## 4. Padrões de Código e Linting

O projeto utiliza o **Ruff** com regras estritas configuradas no `pyproject.toml`. Antes de fazer commits ou alterações:

```powershell
# Checagem de regras e imports
python -m ruff check core interfaces memory tools

# Formatação automática de código
python -m ruff format core interfaces memory tools
```
