# Project AIGIS — Guia de Operação & Setup Local

Este documento descreve como configurar, iniciar e operar o ecossistema completo da **Aigis** localmente no Linux ou Windows.

---

## 1. Pré-Requisitos

- **Sistema Operacional:** Linux (Arch Linux / Ubuntu / Debian) ou Windows 10/11 x64
- **Hardware Recomendado:**
  - GPU AMD Radeon (ex: RX 7600 com ROCm/DirectML/Vulkan) ou NVIDIA com CUDA
  - Microfone calibrado e saída de áudio estéreo
- **Ambiente de Software:**
  - Python 3.10+ gerenciado via Micromamba ou Conda (`aigis`)
  - Docker & Docker Compose (para PostgreSQL com pgvector e Redis)
  - `llama-server` (llama.cpp compilado com suporte a GPU)

---

## 2. Inicialização dos Serviços

### Passo 1: Subir o Banco de Dados e o Barramento (PostgreSQL + pgvector + Redis)
No diretório raiz do projeto:
```bash
docker compose up -d
```
Dois containers serão iniciados:
- `ai_assistant_postgres`: porta **5433** (banco relacional, vetores `pgvector` e timeline).
- `ai_assistant_redis`: porta **6379** (barramento Pub/Sub de baixa latência e estado efêmero).

### Passo 2: Configurar Integração Google Workspace (Opcional, mas Recomendado)
Para habilitar o Google Calendar, Gmail inteligente e Google Drive:
1. Coloque o arquivo `credentials.json` (obtido no Google Cloud Console) na raiz do projeto.
2. Execute o assistente de autenticação OAuth2 local:
   ```bash
   python tools/setup_google_auth.py
   ```
3. O arquivo `token.json` será gerado automaticamente com os escopos autorizados.

### Passo 3: Iniciar o Servidor LLM Local
Em um terminal dedicado, inicie o `llama-server` apontando para o modelo Qwen2.5:
```bash
./start_brain.sh
# Ou no Windows:
# .\start_brain.ps1
```

### Passo 4: Iniciar as Interfaces da Aigis

- **Para iniciar o Bot do Discord (Hub de Canais + Daemons Proativos):**
  ```bash
  python interfaces/discord_bot.py
  ```
- **Para usar o Assistente de Voz no PC (Microfone + Alto-falante):**
  ```bash
  python interfaces/voice_assistant.py
  ```
- **Para o terminal CLI interativo de desenvolvimento:**
  ```bash
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

# Barramento de Eventos Redis
REDIS_URL=redis://localhost:6379/0

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

```bash
# Checagem de regras e imports
python -m ruff check core interfaces memory tools

# Formatação automática de código
python -m ruff format core interfaces memory tools
```
