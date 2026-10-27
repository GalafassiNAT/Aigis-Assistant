# Project AIGIS — Análise de Acoplamento, Extensibilidade & Portabilidade

Este documento avalia o grau de acoplamento da base de código atual e estabelece a facilidade e o custo técnico para:
1. **Escalar ou trocar o modelo de LLM** (ex: Qwen 7B -> Qwen 14B/32B, Llama 3.1, Claude/OpenAI em nuvem).
2. **Adicionar ou alterar ferramentas** (automações, IoT, novos comandos).
3. **Migrar ou aprimorar hardware** (GPU AMD Radeon -> NVIDIA RTX, Linux / ROCm nativo, servidores dedicados).
4. **Composição do Dataset de Treino** (Estratégia bilingue PT-BR + EN).

---

## 1. Escalar ou Trocar o Modelo de LLM

**Facilidade:** ⭐⭐⭐⭐⭐ (Quase Nula / Poucos Cliques)

- **Onde está implementado:** [`core/client.py`](../core/client.py).
- **Como opera:** O cliente utiliza a SDK oficial `AsyncOpenAI`, conectando-se a um endpoint compatível com a API padrão da OpenAI (`http://localhost:8080/v1`).
- **Impacto prático:**
  - Se você quiser trocar o `Qwen2.5-Coder-7B` por um `Qwen2.5-14B`, `Llama-3.1-8B-Instruct` ou `DeepSeek`, você **não altera código**. Apenas altera o comando do `llama-server` apontando para o novo arquivo `.gguf`.
  - Se quiser fazer benchmark com uma API de nuvem (ex: OpenAI GPT-4o ou Groq para testes de latência extrema), basta alterar as variáveis `LLAMA_SERVER_URL` e `LLAMA_MODEL_ALIAS` no [`.env`](../.env).
- **Ponto de Atenção:** Modelos menores (< 7B) tendem a quebrar o parser estrito de JSON de ferramentas. Modelos maiores (14B, 32B) seguem o esquema OpenAPI nativo com ainda mais precisão.

---

## 2. Adicionar, Modificar ou Remover Ferramentas

**Facilidade:** ⭐⭐⭐⭐☆ (Alta / Baixo Acoplamento)

- **Onde está implementado:** [`tools/system.py`](../tools/system.py).
- **Como opera:** A arquitetura adota um padrão de registro unificado via dicionários:
  ```python
  AVAILABLE_TOOLS = {
      "execute_command": execute_command,
      "read_file": read_file,
      "web_search": web_search,
      # novas ferramentas entram aqui
  }
  ```
  E os esquemas que vão para o LLM estão descritos na lista `ALL_TOOL_SCHEMAS`.
- **Como criar uma ferramenta nova:**
  1. Escrever uma função Python com argumentos tipados.
  2. Adicionar a entrada no `ALL_TOOL_SCHEMAS` (formato padrão OpenAPI).
  3. Registrar a função em `AVAILABLE_TOOLS`.
- **Ponto de Melhoria Futura (Desacoplamento Ideal):**
  Transformar em um sistema baseado em *decorators* (ex: `@tool`), eliminando a necessidade de registrar schemas JSON manualmente.

---

## 3. Aprimoramento e Migração de Hardware

**Facilidade:** ⭐⭐⭐⭐☆ (Média-Alta)

### Transição de GPU (AMD Radeon RX 7600 -> NVIDIA RTX)
- **Cérebro (LLM):** O `llama.cpp` roda com backends modulares. Para NVIDIA, apenas troca-se o binário/flags de Vulkan/ROCm para CUDA (`-ngl 99`).
- **Módulo Vocal (RVC + Kokoro):**
  - O pipeline de áudio da Aigis foi intencionalmente construído sobre o **ONNX Runtime com DirectML** (`onnxruntime-directml`), que roda sobre a API DirectX 12 nativa do Windows.
  - O DirectML é **neutro em relação a fabricantes**: ele funciona de forma transparente tanto em AMD (RDNA3) quanto em NVIDIA (GeForce) ou Intel Arc sem necessidade de reescrever código.
  - Caso queira usar CUDA nativo no futuro (ex: Linux ou PyTorch TensorRT), basta alterar a lista de execution providers no [`core/worker_net.py`](../core/worker_net.py):
    ```python
    # De:
    providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    # Para:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ```

---

## 4. Estratégia de Dataset Bilingue (PT-BR + EN)

A inclusão de dados em **Inglês** no LoRA é altamente benéfica pelos seguintes motivos:

1. **Âncora de Raciocínio (Anchor Effect):** O modelo pré-treinado tem 80%+ do seu conhecimento técnico, raciocínio de código e representações lógicas consolidados em inglês. Manter exemplos em inglês preserva o poder de raciocínio lógico intacto e evita *overfitting*.
2. **Datasets Oficiais de Tool Calling:** Os melhores datasets de ferramentas da indústria (como `NousResearch/hermes-function-calling-v1` e `Glaive-Function-Calling-v2`) estão em inglês e cobrem centenas de variações de chamadas de API.
3. **Distribuição Proposta:**
   - **40% Português Conversacional & Persona:** Diálogos naturais em PT-BR, gírias leves, dinamismo e comandos de voz com Natanael.
   - **30% Ferramentas & Raciocínio (Inglês):** Amostras de alta qualidade do Hermes para garantir precisão cirúrgica de chamadas de funções.
   - **30% Ferramentas & Sistema Aigis (Bilingue/PT-BR):** Comandos reais do Windows, rotinas do PowerShell e manipulação de notas do Obsidian.
