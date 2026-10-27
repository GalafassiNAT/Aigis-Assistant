# Plano de Arquitetura e Otimização do Módulo de Voz (Eliminação de Pausas e Destilação RVC-to-TTS)

> **Idioma / Language:** [Português (PT-BR)](VOICE_ARCHITECTURE_PLAN.md) | [English (EN)](VOICE_ARCHITECTURE_PLAN_EN.md)

Este documento detalha o diagnóstico de gargalos de latência do sistema de voz da **Aigis** e fornece o plano completo de engenharia (dividido em 3 fases) para que um agente implementador elimine as pausas entre frases, otimize a execução em GPU (DirectML / Vulkan) e realize a destilação do RVC diretamente em um modelo TTS de estágio único (*Single-Stage TTS*).

---

## 1. Diagnóstico Técnico dos Gargalos Atuais

A análise dos arquivos [`interfaces/voice_assistant.py`](../interfaces/voice_assistant.py), [`core/voice.py`](../core/voice.py), [`core/worker_hubert.py`](../core/worker_hubert.py) e [`core/worker_net.py`](../core/worker_net.py) identificou **4 gargalos críticos** que causam o comportamento de *"Demora -> Responde -> Pausa -> Responde"*:

### 1.1. Bloqueio Síncrono no Loop de Streaming (`sd.wait()`)
* **Onde ocorre:** [`interfaces/voice_assistant.py`](../interfaces/voice_assistant.py) (linhas 133–136) e [`core/voice.py`](../core/voice.py) (método `play()`, linhas 178–185).
* **Causa:** Dentro do `async for token in chat(...):`, quando uma frase termina, o código chama `await synth.speak(frase_para_falar, play_audio=True)`. Por sua vez, `speak()` chama `self.play()`, que executa `sd.play(data, fs)` seguido de **`sd.wait()`** de forma síncrona na thread principal.
* **Efeito:** Enquanto a placa de som está reproduzindo a Frase 1 (ex: 3 segundos), **todo o loop assíncrono do Python fica congelado**. O consumo de tokens do LLM para e a síntese da Frase 2 sequer inicia. A Frase 2 só começa a ser sintetizada quando a Frase 1 termina de tocar, gerando um silêncio de 4 a 6 segundos entre cada frase.

### 1.2. Uso de Rede Externa (`edge-tts`) e I/O de Disco para Português
* **Onde ocorre:** [`core/voice.py`](../core/voice.py) (linhas 140–151).
* **Causa:** O modelo local `Kokoro-ONNX` só está habilitado quando `lang == "en"`. Para `lang == "pt"`, o código realiza uma chamada de rede externa via `edge_tts.Communicate(text, "pt-BR-FranciscaNeural")` aos servidores da Microsoft Azure, salva `temp_base.mp3` no disco, lê o MP3 e o converte para `temp_base.wav` no disco.
* **Efeito:** Adiciona ~800ms a 1.500ms de latência de rede + escrita em disco a cada frase antes mesmo de iniciar o RVC.

### 1.3. Execução do HuBERT e RMVPE na CPU (PyTorch) + Tráfego por Arquivos Temporários
* **Onde ocorre:** [`core/worker_hubert.py`](../core/worker_hubert.py) e [`core/worker_net.py`](../core/worker_net.py).
* **Causa:**
  1. `worker_hubert.py` carrega `HubertModel` em PyTorch **na CPU** (`feats.cpu().numpy()`) e grava `temp_feats.npy` no SSD.
  2. `worker_net.py` carrega o extrator de pitch `RMVPE` (`models/rvc/rmvpe.pt`) em PyTorch **na CPU** (`device="cpu"`), lê `temp_feats.npy` do SSD e grava `output_aigis.wav` no SSD. Apenas o gerador final (`aigis_model.onnx`) roda em DirectML (GPU).
* **Nota:** Os modelos convertidos para ONNX (`models/rvc/vec-768-layer-12.onnx` e `models/rvc/rmvpe.onnx`) **já existem** na pasta `models/rvc/`, mas não estão sendo utilizados pelos workers.

### 1.4. Latência de Pre-fill do LLM no Backend ROCm vs. Vulkan (`start_brain.ps1`)
* **Onde ocorre:** [`start_brain.ps1`](../start_brain.ps1) / [`start_brain.bat`](../start_brain.bat).
* **Causa:** O binário `win-rocm` do `llama-server` na GPU Radeon RX 7600 (`gfx1102`) sem Flash Attention apresenta taxa de avaliação de prompt (*prompt eval*) de ~40 tokens/s, levando ~11 segundos até o primeiro token.
* **Solução:** O binário oficial `llama-b*-bin-win-vulkan-x64` com `-fa` (Flash Attention) e `--cache-reuse 256` atinge 500–700 tokens/s de *prompt eval* na RX 7600, reduzindo o tempo até o primeiro token para **< 400ms**.

---

## 2. Estratégias para Eliminar o RVC (Incluir a Voz da Aigis no TTS)

Para cortar o passo de conversão de voz (RVC) em tempo de execução, existem 3 abordagens técnicas:

| Abordagem | Como Funciona | Latência por Frase | Uso de VRAM | Complexidade de Implementação |
| :--- | :--- | :---: | :---: | :---: |
| **A. Destilação Sintética RVC $\rightarrow$ Piper/VITS ONNX (Recomendada)** | Usa o TTS + RVC atual offline para gerar 1.500 áudios na voz da Aigis e faz Fine-Tuning de um modelo único VITS/Piper exportado para `.onnx`. | **30ms – 60ms** | **< 200 MB** | Média (Treino no Colab ~2h) |
| **B. TTS End-to-End Cross-Lingual (GPT-SoVITS v2/v3)** | Treina o GPT-SoVITS diretamente com 2 a 10 minutos de áudio original da Aigis. Gera PT-BR diretamente com o timbre e emoção da Aigis sem RVC separado. | **250ms – 350ms** | **~2.5 GB** | Baixa (Treino no Colab ~10min) |
| **C. Kokoro-82M Nativo PT-BR + Voice Tensor Blending** | Usa o `Kokoro-ONNX` já instalado com `lang="pt-br"` e otimiza/mistura o vetor de estilo `.bin` (`[510, 1, 256]`) via *Kvoicewalk* sem RVC, ou mantém RVC 100% em ONNX DirectML em RAM. | **80ms (sem RVC) / 350ms (com RVC ONNX)** | **~500 MB** | Baixa (Imediata, sem treino) |

---

## 3. Roteiro de Implementação (Passo a Passo para o Agente)

### FASE 1: Pipeline Assíncrono Produtor-Consumidor (Fim Imediato das Pausas)

**Meta:** Permitir que a Frase $N+1$ seja sintetizada em background enquanto a Frase $N$ está sendo reproduzida na caixa de som.

```mermaid
flowchart LR
    LLM["Task 1: LLM Token Streamer"] -->|Texto da Frase| Q1["sentence_queue (asyncio.Queue)"]
    Q1 -->|Frase| SYNTH["Task 2: TTS/RVC Worker"]
    SYNTH -->|Tuple (np.ndarray PCM, sample_rate)| Q2["audio_queue (asyncio.Queue)"]
    Q2 -->|PCM Array| PLAY["Task 3: Audio Player Thread"]
```

#### Alterações Necessárias:
1. **Em [`core/voice.py`](../core/voice.py):**
   * Criar o método `async def synthesize_array(self, text: str, lang: str = "pt") -> tuple[np.ndarray, int]` que realiza a síntese e retorna `(audio_float32, sample_rate)` diretamente em memória, **sem chamar `self.play()`**.
   * Atualizar `play()` ou criar `async def play_array(self, data: np.ndarray, fs: int)` que rode `sd.play(data, fs); sd.wait()` dentro de `await asyncio.to_thread(...)` para **nunca bloquear o event loop do asyncio**.
2. **Em [`interfaces/voice_assistant.py`](../interfaces/voice_assistant.py):**
   * Substituir a chamada sequencial `await synth.speak(frase, play_audio=True)` dentro do `async for token in chat(...)` por uma arquitetura de 3 Tasks com `asyncio.Queue`:
     * `sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()`
     * `audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue()`
     * **Task Produtora (LLM):** Acumula tokens até pontuação (`.`, `!`, `?`, `:`, `\n`) com tamanho mínimo de 15 caracteres e faz `await sentence_queue.put(frase)`. Ao final do stream, faz `await sentence_queue.put(None)`.
     * **Task Intermediária (Síntese):** Consome `sentence_queue`, chama `await synth.synthesize_array(frase)` e coloca o resultado em `await audio_queue.put((pcm, sr))`. Ao receber `None`, repassa `None` para `audio_queue`.
     * **Task Consumidora (Playback):** Consome `audio_queue` e reproduz via `await synth.play_array(pcm, sr)`.

---

### FASE 2: Eliminação de Rede/Disco no TTS Base e Unificação ONNX em RAM

#### 2.1. Substituir `edge-tts` pelo `Kokoro-ONNX` Nativo em Português
* **Em [`core/voice.py`](../core/voice.py):**
  * Remover a dependência `edge_tts` no fluxo de `lang == "pt"`.
  * Utilizar o `self.kokoro` já carregado em RAM:
    ```python
    # Mistura opcional de timbre (50% pf_dora + 50% af_heart) para manter suavidade da Aigis em PT-BR
    voice_style = 0.5 * self.kokoro.get_voice_style("pf_dora") + 0.5 * self.kokoro.get_voice_style("af_heart")
    samples, sample_rate = self.kokoro.create(text, voice=voice_style, speed=1.05, lang="pt-br")
    ```
  * Isso elimina ~1.2s de download da Microsoft Azure e evita criar `temp_base.mp3`.

#### 2.2. Unificar HuBERT + RMVPE + VITS em DirectML ONNX (Se RVC ainda estiver ativo)
* Substituir [`core/worker_hubert.py`](../core/worker_hubert.py) e [`core/worker_net.py`](../core/worker_net.py) por um único worker que carregue:
  * `models/rvc/vec-768-layer-12.onnx` (em vez de `hubert_base.pt` na CPU)
  * `models/rvc/rmvpe.onnx` (em vez de `rmvpe.pt` na CPU)
  * `models/rvc/aigis_model.onnx`
* Todos rodando com `providers=['DmlExecutionProvider', 'CPUExecutionProvider']` no mesmo processo, trocando buffers de áudio via pipe/memória sem salvar `.wav` ou `.npy` em `temp/`.

#### 2.3. Otimizar `start_brain.ps1` para Vulkan + Flash Attention
* **Em [`start_brain.ps1`](../start_brain.ps1):**
  * Utilizar a build `vulkan` do `llama-server.exe` para a RX 7600.
  * Adicionar os argumentos: `-fa --cache-reuse 256 -b 2048 -ub 512`.

---

### FASE 3: Destilação RVC $\rightarrow$ TTS Único (Aposentando o RVC em Runtime)

Para treinar um modelo TTS único que já fale com a voz da Aigis sem precisar de RVC:

#### Ferramentas Recomendadas
1. **[Piper TTS Training (`rhasspy/piper`)](https://github.com/rhasspy/piper/blob/master/TRAINING.md):** Treinamento VITS em PyTorch com exportação direta para `.onnx` ultrarrápido (~50ms por frase).
2. **[GPT-SoVITS (`RVC-Boss/GPT-SoVITS`)](https://github.com/RVC-Boss/GPT-SoVITS):** Alternativa End-to-End caso se deseje usar os áudios originais em inglês da Aigis para falar português (Cross-Lingual) com alta expressividade via API local (`api_v2.py` em modo streaming).
3. **[Kvoicewalk (`RobViren/kvoicewalk`)](https://github.com/RobViren/kvoicewalk):** Utilitário de evolução de tensores de estilo para encontrar uma matriz `.npy` de 256 dimensões no `Kokoro-82M` que imite um áudio `.wav` alvo da Aigis.

#### Estrutura de Scripts a Criar em `training/tts_distillation/`
1. **`training/tts_distillation/1_generate_corpus.py`**
   * Gera `corpus_ptbr.txt` com 1.500 frases foneticamente ricas em Português (incluindo vocabulário técnico, números, saudações e expressões da persona Aigis).
2. **`training/tts_distillation/2_synthesize_teacher_rvc.py`**
   * Passa as 1.500 frases pelo pipeline atual (Kokoro/Edge-TTS + RVC Aigis), normaliza o áudio em `22050 Hz, 16-bit Mono WAV`, aplica `librosa.effects.trim(top_db=35)` para remover silêncios mortos no início/fim e exporta no formato **LJSpeech** (`dataset/wavs/*.wav` + `dataset/metadata.csv` no padrão `id|texto`).
3. **`training/tts_distillation/3_colab_train_piper.py`**
   * Script automatizado para Google Colab que:
     * Instala `piper_train` e `espeak-ng`.
     * Baixa o checkpoint base em Português (`pt_BR-faber-medium.ckpt`).
     * Faz o pré-processamento fonético (`python3 -m piper_train.preprocess --language pt-br --input-dir dataset/ --output-dir training_dir/ --dataset-format ljspeech --single-speaker --sample-rate 22050`).
     * Executa o fine-tuning (`python3 -m piper_train --dataset-dir training_dir/ --accelerator gpu --devices 1 --batch-size 32 --resume_from_single_speaker_checkpoint pt_BR-faber-medium.ckpt`).
     * Exporta o modelo final (`python3 -m piper_train.export_onnx`) gerando `aigis_tts_ptbr.onnx` e `aigis_tts_ptbr.onnx.json`.
4. **Integração Final em [`core/voice.py`](../core/voice.py):**
   * Carregar `models/piper/aigis_tts_ptbr.onnx` diretamente via `piper-tts` ou `onnxruntime` e desativar os daemons do RVC (`worker_hubert.py` e `worker_net.py`), reduzindo o tempo de síntese de ~5.000ms para **~45ms**.
