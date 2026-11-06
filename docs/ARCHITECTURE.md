# Project AIGIS — Arquitetura e Engenharia do Sistema

A **Aigis** é um sistema de Inteligência Artificial Pessoal Autônomo com presença contínua e multimodal, projetado para operar localmente no hardware de Natanael (Windows 11, AMD Radeon RX 7600), com consciência de estado, memória de longo prazo e capacidade de ação direta no sistema operacional.

---

## 1. Visão Geral da Arquitetura

O sistema é dividido em **4 pilares fundamentais**:

```mermaid
flowchart TD
    subgraph Interfaces ["1. Interfaces de Entrada & Saída"]
        MIC[Microfone Local / STT] --> VA[Voice Assistant]
        DISC[Discord Gateway / Bot] --> DB[Discord Bot]
        CLI[Terminal CLI / Dev]
    end

    subgraph Timeline ["2. Linha do Tempo Unificada"]
        VA <--> ST[Shared Timeline / Postgres]
        DB <--> ST
        ST --> Ctx[Context Manager / Dynamic Prompt]
    end

    subgraph Cognicao ["3. Cérebro & Raciocínio"]
        Ctx --> LLM[llama-server: Qwen2.5-Coder-7B]
        LLM --> Tools[Motor de Execução de Ferramentas]
        Tools --> PS[PowerShell Daemon Persistente]
        Tools --> FS[Filesystem & Web Tools]
    end

    subgraph Memoria ["4. Memória Permanente"]
        LLM -. Gravação em Background .-> MW[Worker de Consolidação]
        MW --> KG[(Grafo de Conhecimento: Tripletos)]
        MW --> VEC[(pgvector: Embeddings FastEmbed)]
        KG -. Recuperação < 5ms .-> Ctx
        VEC -. Similaridade Semântica .-> Ctx
    end

    subgraph Voz ["5. Síntese Vocal Ultrarrápida"]
        LLM --> TTS[Kokoro TTS / ONNX]
        TTS --> RVC[RVC v2 DirectML GPU / AMD RX 7600]
        RVC --> AudioOut[Saída de Áudio / Alto-falante]
    end
```

---

## 2. Módulos do Sistema

### 2.1. Cérebro Local (`core/client.py`)
- **Backend:** `llama-server` (porta `8080`) rodando **Qwen2.5-7B-Instruct** quantizado em `Q4_K_M` com janela de contexto de **8192 tokens**.
- **Modo de Operação:** ReAct (Reasoning + Acting) via OpenAPI Tools nativo e fallback de extração sintática.
- **Roteador e Interceptor Anti-Alucinação:** Impede respostas em texto que afirmem criação, deleção ou modificação de arquivos sem uma chamada de ferramenta verificada (`verified: true`) no turno atual.
- **Hook de Feedback Imediato (`on_tool_start`):** Notifica vocalmente o usuário antes do início de comandos lentos (ex: *"Criando o arquivo..."*, *"Apagando o arquivo..."*, *"Pesquisando na internet..."*), reduzindo a latência percebida para <50ms.

### 2.2. Interfaces (`interfaces/`)
- **Voz no PC (`interfaces/voice_assistant.py`):**
  - **STT (Speech-to-Text):** Faster-Whisper `small` rodando com pré-roll de áudio e parada automática via VAD/energia calibrada.
  - **Áudio de Saída:** Pipeline híbrido em dois estágios: Piper Dii local (PT-BR) / Kokoro ONNX (EN) + Trato Vocal In-VRAM RVC v2 40kHz da Aigis em PyTorch GPU (AMD Radeon RX 7600 via ROCm/Vulkan ou DirectML).
  - **Normalização Fonética de Anglicismos:** Dicionário com mais de 80 termos técnicos (TypeScript, Python, Docker, Desktop, etc.) adaptados em tempo real para pronúncia natural em prosódia brasileira sem distorções robóticas.
- **Discord Bot (`interfaces/discord_bot.py`):**
  - Presença contínua no servidor, sincronização de canais e suporte a anexos de imagem.
  - Sincroniza em tempo real com a Sessão Unificada.

### 2.3. Sessão Unificada & Timeline (`core/timeline.py`)
- **Tabela:** `shared_timeline` no PostgreSQL (porta `5433`).
- **Propósito:** Garante que a Aigis lembre e mantenha coerência entre o que o usuário fala no microfone do PC e o que escreve no Discord.
- **Histórico Estruturado & Sanitização:** Grava chamadas e resultados de ferramentas em metadados JSON, permitindo continuidade de comandos entre múltiplos turnos, e filtra afirmações não verificadas para evitar contaminação de contexto.

### 2.4. Motor de Ferramentas (`tools/system.py`)
- **Manipulação Verificada de Arquivos (`create_file`, `delete_file`):** Criação e exclusão atômica com validação física no disco (`Path.exists()`) e retorno estruturado (`verified: true`).
- **Sessão Persistente PowerShell (Windows):** Daemon interativo com `$ErrorActionPreference = 'Stop'`, UTF-8 forçado e leitor em thread assíncrona com `timeout_seconds` rígido, eliminando travamentos de comandos interativos (ex: `Read-Host`, `pause`) e recuperando o processo em caso de timeout.
- **Subprocess Bash (Linux):** Execução isolada com timeout configurável e truncamento inteligente de saída para proteção do contexto do LLM.
- **Tratamento de Exceções:** Captura real de falhas de caminhos inexistentes e permissões com saída clara de `stderr` e flag `success: false`.

### 2.5. Memória Permanente (`memory/`)
- **Grafo de Conhecimento:** Tripletos relacionais (`sujeito -> predicado -> objeto`) no PostgreSQL para fatos atômicos invariantes (caminhos de projetos, nomes, preferências).
- **Busca Vetorial:** `pgvector` com embeddings `BAAI/bge-small-en-v1.5` gerados via FastEmbed local em CPU/ONNX.
- **Worker Autônomo (`memory/worker.py`):** Processo em background que analisa a timeline e extrai novos fatos sem bloquear a interação do usuário.

---

## 3. Infraestrutura & Portas

| Serviço | Tecnologia | Endereço / Porta | Finalidade |
| :--- | :--- | :--- | :--- |
| **LLM Server** | llama.cpp / llama-server | `http://localhost:8080/v1` | Inferência do modelo neural |
| **Banco de Dados** | PostgreSQL 16 + pgvector | `localhost:5433` (Docker: `ai_assistant_postgres`) | Grafo de conhecimento e timeline |
| **Voz (Kokoro)** | ONNX Runtime | Local / RAM | TTS inicial de alta velocidade |
| **Voz (RVC v2)** | DirectML (`onnxruntime-directml`) | GPU AMD Radeon RX 7600 | Timbre oficial da Aigis |
