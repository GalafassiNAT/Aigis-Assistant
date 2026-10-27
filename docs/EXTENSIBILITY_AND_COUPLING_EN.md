# Project AIGIS — Coupling, Extensibility & Hardware Portability Analysis

This document evaluates the decoupling and technical debt of the current codebase, defining how seamlessly the system can:
1. **Scale or swap the LLM engine** (e.g., Qwen 7B -> Qwen 14B/32B, Llama 3.1, cloud APIs).
2. **Add, alter, or remove tools** (operating system actions, web crawlers, IoT).
3. **Upgrade or migrate hardware** (AMD Radeon -> NVIDIA RTX, Windows -> Linux / native ROCm).
4. **Bilingual Training Strategy** (Balancing PT-BR conversational persona with English reasoning benchmarks).

---

## 1. Scaling or Swapping the LLM Engine

**Portability:** ⭐⭐⭐⭐⭐ (Seamless / Zero Code Changes)

- **Implementation:** [`core/client.py`](../core/client.py).
- **Architecture:** Standardized on the official `AsyncOpenAI` SDK pointing to any OpenAI-compatible HTTP inference endpoint (`http://localhost:8080/v1`).
- **Practical Impact:**
  - Upgrading from `Qwen2.5-Coder-7B` to a larger 14B/32B variant requires **zero python code modifications**—only pointing `llama-server.exe` to the new `.gguf` weights.
  - Benchmarking with cloud engines (OpenAI GPT-4o, Groq, or DeepSeek) requires only changing `LLAMA_SERVER_URL` and `LLAMA_MODEL_ALIAS` inside [`.env`](../.env).
- **Caveats:** Models below 7B parameters may suffer from schema compliance degradation. Larger models (14B+) strictly adhere to OpenAPI specifications.

---

## 2. Tool Extensibility & Maintenance

**Portability:** ⭐⭐⭐⭐☆ (High / Decoupled)

- **Implementation:** [`tools/system.py`](../tools/system.py).
- **Architecture:** Function-dispatch table pattern:
  ```python
  AVAILABLE_TOOLS = {
      "execute_command": execute_command,
      "read_file": read_file,
      "web_search": web_search,
      # new tools registered here
  }
  ```
  Accompanied by OpenAPI schemas inside `ALL_TOOL_SCHEMAS`.
- **Workflow to add new capabilities:**
  1. Define a standard Python function with typed parameters.
  2. Append its JSON schema to `ALL_TOOL_SCHEMAS`.
  3. Map the function to `AVAILABLE_TOOLS`.
- **Future Roadmap:** Refactor into a decorator-based registration pattern (`@tool`) to auto-generate OpenAPI schemas from docstrings.

---

## 3. Hardware Upgrades & Cross-Vendor Migration

**Portability:** ⭐⭐⭐⭐☆ (Medium-High)

### GPU Migration (AMD Radeon RX 7600 -> NVIDIA GeForce RTX)
- **Cognition (LLM):** `llama.cpp` is inherently cross-platform. Switching to NVIDIA simply involves running a CUDA-compiled binary with `-ngl 99`.
- **Vocal Pipeline (Kokoro TTS + RVC v2):**
  - Audio transformation is built entirely on top of **ONNX Runtime with DirectML** (`onnxruntime-directml`), utilizing Microsoft DirectX 12.
  - DirectML is **vendor-agnostic**: it executes identically across AMD (RDNA), NVIDIA (GeForce/RTX), and Intel Arc GPUs without changes to the underlying models or scripts.
  - If native Linux/CUDA execution is desired in the future, updating execution providers inside [`core/worker_net.py`](../core/worker_net.py) takes two lines:
    ```python
    # From:
    providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    # To:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ```

---

## 4. Bilingual Training Strategy (PT-BR + EN)

Integrating English datasets into the LoRA fine-tuning corpus offers critical advantages:

1. **Reasoning Anchor:** The base model possesses over 80% of its coding logic, tool schema knowledge, and structural representations in English. Preserving English examples acts as a regularizer, preventing *catastrophic forgetting* and overfitting.
2. **Industry Tool-Calling Benchmarks:** The highest quality function-calling datasets (`NousResearch/hermes-function-calling-v1`, `Glaive-Function-Calling-v2`) are native English.
3. **Target Blending Distribution:**
   - **40% PT-BR Conversational Persona:** Organic Portuguese speech, emotional resonance, concise voice acknowledgments with Natanael.
   - **30% English Tool Calling & Reasoning:** High-fidelity samples from Hermes to guarantee zero tool-calling regressions.
   - **30% Domain-Specific Aigis Scenarios (Bilingual):** Concrete Windows PowerShell commands, Obsidian file paths, and mixed-language terminal interaction.
