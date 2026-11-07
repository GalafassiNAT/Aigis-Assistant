# Project AIGIS — System Architecture & Engineering

**Aigis** is an autonomous, multimodal Personal Artificial Intelligence system designed to operate locally on Natanael's hardware (Windows 11, AMD Radeon RX 7600), featuring state awareness, persistent long-term memory, and direct operating system actuation capabilities.

---

## 1. Architectural Overview

The system is structured around **6 fundamental pillars**:

```mermaid
flowchart TD
    subgraph Interfaces ["1. Input & Output Interfaces"]
        MIC[Local Microphone / STT] --> VA[Voice Assistant]
        DISC[Discord Gateway / Bot] --> DB[Discord Bot]
        CLI[Developer CLI Terminal]
    end

    subgraph Bus ["2. Real-Time Event Bus & State Sync"]
        REDIS[(Redis 7: Pub/Sub & Ephemeral State)]
        VA <--> REDIS
        DB <--> REDIS
        PH[Proactive Life Hub] <--> REDIS
    end

    subgraph Timeline ["3. Unified Shared Timeline"]
        VA <--> ST[Shared Timeline / Postgres]
        DB <--> ST
        ST --> Ctx[Context Manager / Dynamic Prompt]
    end

    subgraph Cognition ["4. Brain & Reasoning Engine"]
        Ctx --> LLM[llama-server: Qwen2.5-7B-Instruct]
        Persona[(Persona State: data/persona_state.json)] <--> Ctx
        LLM --> Tools[Tool Execution Engine]
        Tools --> PS[PowerShell / Bash Runners]
        Tools --> FS[Filesystem & Web Tools]
        Tools --> GSuite[Google Workspace: Calendar, Gmail, Drive]
    end

    subgraph Memory ["5. Permanent Memory & Reflection"]
        LLM -. Background Daemon (60s) .-> MW[Cognitive Reflection Worker]
        MW --> KG[(Knowledge Graph: Triplets)]
        MW --> VEC[(pgvector: FastEmbed Embeddings)]
        MW --> EP[(Autobiographical Memory: user_profiles & episodic_memories)]
        MW --> DC[Dataset Curator: SFT / DPO]
        DC --> DTR[(data/training: SFT & DPO jsonl)]
        MW -. Evolves .-> Persona
        KG -. < 5ms Retrieval .-> Ctx
        VEC -. Semantic Similarity .-> Ctx
        EP -. Profiles & Episodes .-> Ctx
    end

    subgraph Voice ["6. Ultra-low Latency Voice Synthesis"]
        LLM --> TTS[Piper Dii PT-BR / Kokoro EN]
        TTS --> RVC[RVC v2 In-VRAM GPU / AMD RX 7600]
        RVC --> AudioOut[Audio Playback / Speakers]
    end
```

---

## 2. Core Modules

### 2.1. Local Brain (`core/client.py`)
- **Backend:** `llama-server` (port `8080`) serving **Qwen2.5-7B-Instruct** quantized in `Q4_K_M` with an **8192-token** context window.
- **Reasoning Mode:** ReAct (Reasoning + Acting) via native OpenAPI Tool Schemas with regex syntax extraction fallbacks.
- **Context-Aware Tool Router (`should_enable_tools`):** Evaluates user prompts and conversation turns to enable tools only when real action intent exists. Handles semantic negations (*"there is no email"*, *"just asking for opinion"*) to keep tools disabled and stream responses immediately in ~1s.
- **Strictly Scoped Anti-Hallucination Interceptor:** Blocks text responses simulating actions without corresponding tool executions, strictly scoped to the exact domain requested by the user to avoid false-positive locks during conceptual discussions.
- **Channel Differentiation (Discord vs PC Voice):** Adapts response density dynamically by interface — comprehensive, analytical, structured plans in Discord (text) and ultra-concise 1-sentence cues for PC voice loops.
- **Dynamic Cognitive Injection Preserving KV Cache:** The system prompt (`prompts/system.md`) remains 100% static to maximize prefix KV cache reuse in `llama-server`. Dynamic cognitive context (living persona mood, user profile traits, recent episodic memory summaries, and semantic memories) is appended strictly to the tail of the final user message (`obter_contexto_cognitivo_completo()`).
- **Cognitive Co-Pilot & Brainstorming Partner:** Explicit directives in the system prompt establishing Aigis as an active intellectual co-pilot — constructively challenging assumptions, proposing architectural alternatives, and collaborating as a thought partner.
- **Immediate Tool Audio Hook (`on_tool_start`):** Vocally cues the user before starting long-running actions (e.g., *"Creating file..."*, *"Reading email..."*), reducing perceived latency to <50ms.

### 2.2. Interfaces (`interfaces/`)
- **PC Voice Assistant (`interfaces/voice_assistant.py`):**
  - **STT (Speech-to-Text):** Faster-Whisper `small` running on CPU/INT8 with calibrated energy VAD and automatic silence cutoffs.
  - **Audio Pipeline:** Hybrid two-stage synthesis: Piper Dii (PT-BR) / Kokoro ONNX (EN) + In-VRAM RVC v2 40kHz PyTorch GPU vocal tract on the AMD Radeon RX 7600 GPU.
  - **Technical Phonetic Normalization:** Over 80 anglicisms phonetically transcribed for natural Portuguese prosody without robotic glitches.
- **Discord Bot (`interfaces/discord_bot.py`):**
  - Continuous server presence with dedicated functional channels: `#aigis-agenda`, `#aigis-inbox`, `#aigis-feed`, `#aigis-status`.
  - Real-time Redis synchronization and multi-turn attachment handling.

### 2.3. Unified Session & Timeline (`core/timeline.py`)
- **Storage:** `shared_timeline` table in PostgreSQL (port `5433`).
- **Functionality:** Ensures memory consistency between PC microphone voice commands and Discord chat interactions.
- **Tool Trace Persistence:** Retains full `tool_calls` and execution results within message metadata, enabling context continuity across multiple conversational turns.

### 2.4. Proactive Life Hub & Event Bus (`core/proactive_hub.py` & `core/bus.py`)
- **Redis Pub/Sub Event Bus:** Instantaneous (<1ms) event synchronization across Discord, PC voice assistant, and background daemons.
- **Agenda Reminders:** Continuous 60-second polling notifying `#aigis-agenda` 15 to 30 minutes ahead of scheduled events with owner pings.
- **Intelligent Gmail Triage:**
  - Automated newsletter and marketing blast pruning via RFC headers (`List-Unsubscribe`, `List-Id`) and heuristic sender analysis.
  - Prioritized visual alerts in `#aigis-inbox` with owner mentions for urgent emails.
  - **Persistent Disk Deduplication (`data/notified_emails.json`):** Bulletproof persistence preventing re-notifications across bot reboots.

### 2.5. Tool Engine (`tools/system.py`)
- **Google Calendar (`create_calendar_event`, `delete_calendar_event`):** Semantic weekday resolution, recurring events (`RRULE:FREQ=WEEKLY;BYDAY=...`), thematic category color coding (`colorId`), and duplicate-safe mass deletion.
- **Verified Email Reading (`read_email_body`):** True plaintext MIME extraction from Gmail via ID or flexible topic/urgency queries, protected by anti-hallucination guardrails.
- **Google Drive (`search_google_drive`, `read_google_drive_file`):** Search and read documents and spreadsheets directly within the user workspace.
- **Filesystem Tools (`create_file`, `delete_file`):** Atomic verified file operations with physical disk confirmation (`verified: true`).
- **Persistent Shell Runners:** Interactive PowerShell daemon (Windows) and isolated Bash runner (Linux) with strict timeout deadlines.

### 2.6. Permanent & Autobiographical Memory (`memory/`)
- **Knowledge Graph:** Relational triplets (`subject -> predicate -> object`) in PostgreSQL for atomic, invariant facts about the world and the user.
- **Vector Search:** `pgvector` index with dense embeddings computed locally via FastEmbed ONNX for contextual semantic recall.
- **User Profiling (`user_profiles`):** Relational table keyed on `trait_key` capturing Natanael's ongoing sociopsychological profile (communication styles, late-night working habits, academic schedule, technical preferences, active projects) with confidence scoring (`confidence: 0.0 - 1.0`).
- **Episodic Memory (`episodic_memories`):** Structured log of past memorable sessions, architectural milestones, and personal moments, indexed by timestamp and subjective importance (`importance: 1 - 5`).

### 2.7. Living Persona State (`core/persona.py`)
- **State File:** `data/persona_state.json`.
- **Organic Evolution:** Models Aigis's real-time emotional and relational disposition:
  - `mood` (e.g., focused, cheerful, complicit, reflective, caring).
  - `energy` (0.0 to 1.0).
  - `affinity_level` (0.0 to 1.0, tracking intellectual bond and rapport).
  - `shared_vocabulary` (insider shorthand, codebase nicknames, and shared idioms formed over time).
  - `behavioral_adaptations` (learned behavioral patterns, e.g., staying extra concise during late-night debugging).

### 2.8. Cognitive Reflection Worker & Dataset Curator (`memory/worker.py` & `memory/dataset_curator.py`)
- **Continuous Reflection Loop:** Runs as a background daemon every 60 seconds within the Discord Bot process (`_cognitive_daemon`).
- **Unified Single-Pass LLM Extraction:** Evaluates unreflected `shared_timeline` batches in a single Qwen2.5 prompt, extracting concurrently:
  1. Atomic facts for the Knowledge Graph.
  2. Incremental user profile traits (`user_profiles`).
  3. Key interaction episodes (`episodic_memories`).
  4. Persona mood adjustments and shared vocabulary (`data/persona_state.json`).
  5. Detection of user corrections or negative feedback on previous assistant actions.
- **Continuous Dataset Curation for LoRA Alignment:**
  - **SFT (`data/training/aigis_sft.jsonl`):** High-quality multi-turn dialogues preserved in ChatML format.
  - **DPO (`data/training/aigis_dpo.jsonl`):** Preference pairs (`prompt`, `chosen`, `rejected`) automatically extracted whenever the user reprimands or corrects an assistant response.

### 2.9. Autonomous Initiative Engine (`core/proactive_hub.py`)
- **Spontaneous Initiative:** Aigis takes action proactively without waiting for user commands.
- **Late-Night Sleep & Health Guardian:** Between 23:30 and 04:00, if active interactions are detected on the timeline, Aigis proactively reaches out with a caring reminder advising rest.
- **Post-Habit Check-Ins:**
  - ~09:00 (weekdays): Encouraging check-in regarding morning Mandarin studies.
  - ~22:00: Progress check on daily LeetCode problem solving.
- **Anti-Spam Locks via Redis:** Distributed TTL keys (`lock:initiative:sleep_reminder`, `lock:initiative:habit_mandarin`, etc.) ensure non-intrusive, single-fire execution.

---

## 3. Infrastructure & Services

| Service | Technology | Address / Port | Role |
| :--- | :--- | :--- | :--- |
| **LLM Server** | llama.cpp / llama-server | `http://localhost:8080/v1` | Neural model inference (Qwen2.5-7B) |
| **Database** | PostgreSQL 16 + pgvector | `localhost:5433` (Docker: `ai_assistant_postgres`) | Knowledge graph, vectors, and shared timeline |
| **Event Bus & State** | Redis 7 | `localhost:6379` (Docker: `ai_assistant_redis`) | Real-time Pub/Sub and state management |
| **Google Workspace** | Google APIs OAuth2 | Local `token.json` | Calendar, Gmail, and Drive integrations |
| **Voice (Piper / Kokoro)** | ONNX Runtime | Local / RAM | High-speed base phonetic TTS |
| **Voice (RVC v2)** | PyTorch DirectML / ROCm | AMD Radeon RX 7600 GPU | Official Aigis vocal timbre (40kHz) |
