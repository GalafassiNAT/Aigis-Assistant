# Voice Module Architecture & Optimization Plan (Eliminating Pauses & RVC-to-TTS Distillation)

> **Idioma / Language:** [Português (PT-BR)](VOICE_ARCHITECTURE_PLAN.md) | [English (EN)](VOICE_ARCHITECTURE_PLAN_EN.md)

This document details the latency bottleneck diagnosis of the **Aigis** voice subsystem and provides a complete 3-phase engineering plan for an implementation agent to eliminate inter-sentence pauses, optimize GPU execution (DirectML / Vulkan), and distill the RVC voice directly into a single-stage TTS model.

---

## 1. Technical Diagnosis of Current Bottlenecks

An inspection of [`interfaces/voice_assistant.py`](../interfaces/voice_assistant.py), [`core/voice.py`](../core/voice.py), [`core/worker_hubert.py`](../core/worker_hubert.py), and [`core/worker_net.py`](../core/worker_net.py) identified **4 critical bottlenecks** responsible for the *"Wait -> Speak -> Pause -> Speak"* behavior:

### 1.1. Synchronous Playback Blocking in the Streaming Loop (`sd.wait()`)
* **Location:** [`interfaces/voice_assistant.py`](../interfaces/voice_assistant.py) (lines 133–136) and [`core/voice.py`](../core/voice.py) (`play()` method, lines 178–185).
* **Cause:** Inside `async for token in chat(...):`, when a sentence ends, the loop calls `await synth.speak(frase_para_falar, play_audio=True)`. Inside `speak()`, `self.play()` runs `sd.play(data, fs)` followed by **`sd.wait()`** synchronously on the main thread.
* **Effect:** While Sentence 1 is playing out loud (e.g., 3 seconds), **the entire Python asyncio event loop freezes**. LLM token consumption halts and Sentence 2 synthesis does not even start until Sentence 1 finishes playing—causing a 4–6 second silence between every sentence.

### 1.2. External Network Calls (`edge-tts`) & Disk I/O for Portuguese
* **Location:** [`core/voice.py`](../core/voice.py) (lines 140–151).
* **Cause:** Local `Kokoro-ONNX` is only enabled when `lang == "en"`. For `lang == "pt"`, the code makes an external network request via `edge_tts.Communicate(text, "pt-BR-FranciscaNeural")` to Microsoft Azure servers, saves `temp_base.mp3` to disk, reads it back, and writes `temp_base.wav` to disk.
* **Effect:** Adds ~800ms to 1,500ms of network + disk latency per sentence before RVC even starts.

### 1.3. CPU-Bound HuBERT & RMVPE Execution + Temporary File IPC
* **Location:** [`core/worker_hubert.py`](../core/worker_hubert.py) and [`core/worker_net.py`](../core/worker_net.py).
* **Cause:**
  1. `worker_hubert.py` runs `HubertModel` in PyTorch **on CPU** (`feats.cpu().numpy()`) and writes `temp_feats.npy` to the SSD.
  2. `worker_net.py` runs the `RMVPE` pitch extractor (`models/rvc/rmvpe.pt`) in PyTorch **on CPU** (`device="cpu"`), reads `temp_feats.npy` from disk, and writes `output_aigis.wav` to disk. Only the final generator (`aigis_model.onnx`) runs on DirectML (GPU).
* **Note:** Converted ONNX models (`models/rvc/vec-768-layer-12.onnx` and `models/rvc/rmvpe.onnx`) **already exist** in `models/rvc/` but are currently unused by the workers.

### 1.4. LLM Pre-fill Latency on ROCm vs. Vulkan Backend (`start_brain.ps1`)
* **Location:** [`start_brain.ps1`](../start_brain.ps1) / [`start_brain.bat`](../start_brain.bat).
* **Cause:** Running the `win-rocm` build of `llama-server` on the Radeon RX 7600 (`gfx1102`) without Flash Attention yields ~40 tokens/s prompt evaluation, taking ~11 seconds to produce the first token.
* **Solution:** Switching to the official `llama-b*-bin-win-vulkan-x64` build with `-fa` (Flash Attention) and `--cache-reuse 256` reaches 500–700 tokens/s prompt evaluation on the RX 7600, dropping Time-to-First-Token to **< 400ms**.

---

## 2. Strategies to Eliminate RVC (Embedding Aigis's Voice into TTS)

To cut out the runtime RVC conversion step, there are 3 viable technical strategies:

| Strategy | How It Works | Latency per Sentence | VRAM Usage | Implementation Effort |
| :--- | :--- | :---: | :---: | :---: |
| **A. Synthetic RVC $\rightarrow$ Piper/VITS ONNX Distillation (Recommended)** | Uses the current TTS + RVC pipeline offline as a "Teacher" to generate 1,500 audio clips in Aigis's voice, then fine-tunes a single-stage VITS/Piper `.onnx` model. | **30ms – 60ms** | **< 200 MB** | Medium (~2h Colab training) |
| **B. End-to-End Cross-Lingual TTS (GPT-SoVITS v2/v3)** | Fine-tunes GPT-SoVITS directly on 2–10 minutes of original Aigis audio. Synthesizes PT-BR directly with Aigis's timbre and emotion without a separate RVC stage. | **250ms – 350ms** | **~2.5 GB** | Low (~10min Colab training) |
| **C. Native PT-BR Kokoro-82M + Voice Tensor Blending** | Uses the already-installed `Kokoro-ONNX` with `lang="pt-br"` and blends/optimizes the `.bin` style vector (`[510, 1, 256]`) via *Kvoicewalk* without RVC, or keeps RVC 100% in ONNX DirectML in RAM. | **80ms (no RVC) / 350ms (ONNX RVC)** | **~500 MB** | Low (Immediate, zero training) |

---

## 3. Implementation Roadmap (Step-by-Step for the Agent)

### PHASE 1: Producer-Consumer Async Pipeline (Immediate End to Inter-Sentence Pauses)

**Goal:** Synthesize Sentence $N+1$ in the background while Sentence $N$ is playing out loud.

```mermaid
flowchart LR
    LLM["Task 1: LLM Token Streamer"] -->|Sentence String| Q1["sentence_queue (asyncio.Queue)"]
    Q1 -->|Sentence| SYNTH["Task 2: TTS/RVC Worker"]
    SYNTH -->|Tuple (np.ndarray PCM, sample_rate)| Q2["audio_queue (asyncio.Queue)"]
    Q2 -->|PCM Array| PLAY["Task 3: Audio Player Thread"]
```

#### Required Changes:
1. **In [`core/voice.py`](../core/voice.py):**
   * Create `async def synthesize_array(self, text: str, lang: str = "pt") -> tuple[np.ndarray, int]` that synthesizes and returns `(audio_float32, sample_rate)` in memory **without calling `self.play()`**.
   * Update `play()` or add `async def play_array(self, data: np.ndarray, fs: int)` that runs `sd.play(data, fs); sd.wait()` inside `await asyncio.to_thread(...)` so it **never blocks the asyncio event loop**.
2. **In [`interfaces/voice_assistant.py`](../interfaces/voice_assistant.py):**
   * Replace the sequential `await synth.speak(frase, play_audio=True)` inside `async for token in chat(...)` with 3 concurrent tasks connected via `asyncio.Queue`:
     * `sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()`
     * `audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue()`
     * **Producer Task (LLM):** Accumulates tokens until punctuation (`.`, `!`, `?`, `:`, `\n`) with a minimum length of 15 characters and calls `await sentence_queue.put(frase)`. Pushes `None` when the stream ends.
     * **Synthesizer Task:** Reads from `sentence_queue`, runs `await synth.synthesize_array(frase)`, and pushes `(pcm, sr)` to `audio_queue`. Forwards `None` upon completion.
     * **Player Task:** Reads from `audio_queue` and plays audio sequentially via `await synth.play_array(pcm, sr)`.

---

### PHASE 2: Removing Network/Disk TTS & Unifying ONNX in RAM

#### 2.1. Replace `edge-tts` with Native Portuguese `Kokoro-ONNX`
* **In [`core/voice.py`](../core/voice.py):**
  * Remove `edge_tts` from the `lang == "pt"` path.
  * Use the already-loaded `self.kokoro` instance in RAM:
    ```python
    voice_style = 0.5 * self.kokoro.get_voice_style("pf_dora") + 0.5 * self.kokoro.get_voice_style("af_heart")
    samples, sample_rate = self.kokoro.create(text, voice=voice_style, speed=1.05, lang="pt-br")
    ```

#### 2.2. Unify HuBERT + RMVPE + VITS on DirectML ONNX (If RVC Remains Active)
* Replace [`core/worker_hubert.py`](../core/worker_hubert.py) and [`core/worker_net.py`](../core/worker_net.py) with a single unified worker loading:
  * `models/rvc/vec-768-layer-12.onnx` (replaces CPU `hubert_base.pt`)
  * `models/rvc/rmvpe.onnx` (replaces CPU `rmvpe.pt`)
  * `models/rvc/aigis_model.onnx`
* Run all three on `providers=['DmlExecutionProvider', 'CPUExecutionProvider']` in a single process, passing audio arrays in memory instead of writing `.wav` and `.npy` files to `temp/`.

#### 2.3. Optimize `start_brain.ps1` for Vulkan + Flash Attention
* **In [`start_brain.ps1`](../start_brain.ps1):**
  * Use the `vulkan` build of `llama-server.exe` for the RX 7600.
  * Add flags: `-fa --cache-reuse 256 -b 2048 -ub 512`.

---

### PHASE 3: RVC $\rightarrow$ Single-Stage TTS Distillation (Retiring Runtime RVC)

To train a single-stage TTS model with Aigis's voice built in:

#### Recommended Tools
1. **[Piper TTS Training (`rhasspy/piper`)](https://github.com/rhasspy/piper/blob/master/TRAINING.md):** VITS PyTorch training with direct export to ultra-fast `.onnx` (~50ms per sentence).
2. **[GPT-SoVITS (`RVC-Boss/GPT-SoVITS`)](https://github.com/RVC-Boss/GPT-SoVITS):** End-to-end alternative to use Aigis's original English voice lines to speak fluent Portuguese (Cross-Lingual) via its local streaming API (`api_v2.py`).
3. **[Kvoicewalk (`RobViren/kvoicewalk`)](https://github.com/RobViren/kvoicewalk):** Style tensor optimizer to evolve a 256-dim `.npy` voice vector for `Kokoro-82M` matching an Aigis reference `.wav`.

#### Scripts to Create in `training/tts_distillation/`
1. **`training/tts_distillation/1_generate_corpus.py`**
   * Generates `corpus_ptbr.txt` with 1,500 phonetically balanced Portuguese sentences (technical vocabulary, numbers, greetings, and Aigis persona responses).
2. **`training/tts_distillation/2_synthesize_teacher_rvc.py`**
   * Runs the 1,500 sentences through the current Teacher pipeline (Kokoro/Edge-TTS + Aigis RVC), normalizes audio to `22050 Hz, 16-bit Mono WAV`, trims leading/trailing silence (`librosa.effects.trim(top_db=35)`), and exports in **LJSpeech** format (`dataset/wavs/*.wav` + `dataset/metadata.csv` formatted as `id|text`).
3. **`training/tts_distillation/3_colab_train_piper.py`**
   * Automated Google Colab script that:
     * Installs `piper_train` and `espeak-ng`.
     * Downloads the base Portuguese checkpoint (`pt_BR-faber-medium.ckpt`).
     * Preprocesses phonemes (`python3 -m piper_train.preprocess --language pt-br --input-dir dataset/ --output-dir training_dir/ --dataset-format ljspeech --single-speaker --sample-rate 22050`).
     * Fine-tunes the model (`python3 -m piper_train --dataset-dir training_dir/ --accelerator gpu --devices 1 --batch-size 32 --resume_from_single_speaker_checkpoint pt_BR-faber-medium.ckpt`).
     * Exports the final ONNX model (`python3 -m piper_train.export_onnx`) producing `aigis_tts_ptbr.onnx` and `aigis_tts_ptbr.onnx.json`.
4. **Final Integration in [`core/voice.py`](../core/voice.py):**
   * Load `models/piper/aigis_tts_ptbr.onnx` directly via `piper-tts` or `onnxruntime` and disable the RVC daemons (`worker_hubert.py` and `worker_net.py`), cutting synthesis time from ~5,000ms down to **~45ms**.
