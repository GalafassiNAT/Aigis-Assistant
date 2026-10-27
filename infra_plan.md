# Planejamento de Arquitetura: Agente Autônomo Local com Memória de Grafo e Storage (Estilo "Jarvis/Neuro-Sama")

## 1. Visão Geral do Sistema
* **Objetivo:** Construir um assistente autônomo local integrado ao desktop e ao Discord, capaz de interagir em tempo real, executar ferramentas (*tool calling*), ler tela/código e manter memória de longo prazo persistente.
* **Hardware Alvo:** AMD Ryzen 7 5700X3D + AMD Radeon RX 7600 (8 GB VRAM) + 16 GB RAM + SSD NVMe.
* **Cérebro (LLM):** `llama-server` (llama.cpp) rodando via backend **ROCm**, servindo modelos quantizados (`Q4_K_M`) em `localhost:8080/v1` a ~40 tokens/s.

---

## 2. Pilha Tecnológica (Tech Stack)
* **Orquestrador Central:** Python 3.10+ (gerenciando loops assíncronos, chamadas de API e threads).
* **Interface de Comunicação:** `discord.py` (para interação via chat privado/servidor local).
* **Camada de Memória de Longo Prazo:** 
  * **Vetor/Busca Semântica:** SQLite + extensões locais ou ChromaDB (armazenado no SSD).
  * **Grafo de Conhecimento:** FalkorDB ou NetworkX persistido em JSON/SQLite para mapear entidades, relações e fatos sobre o usuário e projetos.
* **Cache de Contexto:** Gerenciamento dinâmico de janelas de contexto (`ctx-size = 16384` com quantização KV `q8_0`).
* **Visão e Automação:** `pyautogui` / `mss` (captura de tela sob demanda) + bibliotecas nativas do Windows para execução de comandos (`subprocess`).

---

## 3. Fases de Implementação para o Agente (Roadmap)

### Fase 1: Fundação do Servidor e API (Concluído / Base)
* [x] Configurar `llama-server` com ROCm (`--device ROCm0`).
* [x] Otimizar parâmetros de inicialização (`flash-attn = on`, `cache-type-k = q8_0`, `cache-type-v = q8_0`, `ubatch-size = 256`).
* [x] Subir o modelo `Qwen2.5-Coder-7B-Instruct` com 16K de contexto mantendo ~40 t/s.

### Fase 2: O Orquestrador em Python (O "Loop Principal")
* [ ] Criar o script central (`main.py`) que conecta via cliente assíncrono OpenAI (`openai` SDK apontando para `http://localhost:8080/v1`).
* [ ] Implementar o sistema de prompt de sistema dinâmico (comportamento conversacional, focado em respostas curtas e diretas, sem excesso de markdown se for focado em voz/chat rápido).
* [ ] Configurar tratamento de histórico de curto prazo (sliding window na RAM).

### Fase 3: Arquitetura de Memória de Longo Prazo (Grafo + SSD)
* [ ] **Extração de Fatos (Memory Extractor):** Toda vez que uma sessão termina ou a cada X interações, o agente roda um prompt de fundo leve para extrair entidades e relações da conversa (Ex: `[Natanael] -> [usa] -> [Arch Linux]`).
* [ ] **Persistência em Grafo:** Salvar as relações estruturadas em um arquivo local no SSD (`memory_graph.json` ou banco SQLite local).
* [ ] **Recuperação Contextual (Graph RAG):** Quando o usuário fizer uma pergunta, o script faz uma varredura nas entidades mencionadas e injeta apenas os nós relevantes do grafo no prompt do LLM.

### Fase 4: Integração com o Discord e Ferramentas (Tools)
* [ ] Configurar o bot do Discord para escutar mensagens em DMs ou canais específicos.
* [ ] Implementar o mecanismo de *Function Calling* nativo do `llama-server`:
  * `executar_comando(cmd)`: Roda scripts ou comandos no terminal com restrições de segurança.
  * `ler_arquivo_projeto(path)`: Lê arquivos locais de código para o agente analisar.
  * `consultar_grafo_memoria(query)`: Busca fatos passados no armazenamento interno.

---

## 4. Configuração de Referência do Servidor (`config.ini`)
```ini
version = 1

[*]
device = ROCm0
n-gpu-layers = 99
parallel = 1
flash-attn = on
cache-type-k = q8_0
cache-type-v = q8_0
ubatch-size = 256
cache-ram = 0
no-cache-idle-slots = true
offline = true
dedup-cache-models = true

[qwen-coder]
hf-repo = bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M
alias = qwen-coder
ctx-size = 16384
load-on-startup = true