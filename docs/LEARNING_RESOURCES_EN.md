# Project AIGIS — Study Guide, Research Papers & Recommended Readings

This document compiles the foundational **research papers**, **reference books**, and **engineering guides** to master the technologies driving this project: Transformer architectures, Parameter-Efficient Fine-Tuning (PEFT/LoRA), quantization, agentic tool execution, and local LLM systems engineering.

---

## 1. Seminal Research Papers

### 1.1. Parameter-Efficient Fine-Tuning (PEFT / LoRA / QLoRA)
1. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
   - *Core Takeaway:* Introduced the low-rank matrix decomposition technique $W + B \times A$. Essential reading for understanding why training less than 1% of parameters achieves full fine-tuning performance.
   - *Link:* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
2. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
   - *Core Takeaway:* Introduced 4-bit NormalFloat (`NF4`), double quantization, and paged optimizers, making it feasible to fine-tune 7B–65B models on consumer GPUs (such as Colab's 16GB T4).
   - *Link:* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
3. **Instruction Tuning with GPT-4** (Peng et al., 2023)
   - *Core Takeaway:* How structured multi-turn synthetic instruction datasets guide persona alignment and reduce base model hallucinations.
   - *Link:* [arXiv:2304.03277](https://arxiv.org/abs/2304.03277)

### 1.2. Transformer Foundations & Base Models
4. **Attention Is All You Need** (Vaswani et al., 2017)
   - *Core Takeaway:* The groundbreaking paper that introduced the Transformer architecture, scaled dot-product attention, and multi-head projection tensors.
   - *Link:* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
5. **Qwen2.5 Technical Report** (Qwen Team, 2024)
   - *Core Takeaway:* Structural breakdown of the base model powering Aigis (Qwen2.5-Coder), detailing RoPE embeddings, SwiGLU activation, and context scaling.
   - *Link:* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

### 1.3. Autonomous Agents & Tool Use
6. **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022)
   - *Core Takeaway:* The paradigm implemented in `core/client.py` (interleaving Thought -> Action/Tool -> Observation -> Response).
   - *Link:* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
7. **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023)
   - *Core Takeaway:* Autonomous self-supervised learning for API call triggers.
   - *Link:* [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)

---

## 2. Recommended Books

### 2.1. The Ultimate Practical Book
- **"Build a Large Language Model (from Scratch)"** — *Sebastian Raschka (Manning Publications, 2024)*
  - **Why Read:** Hands down the single best book for practical understanding. Raschka builds a complete GPT-like LLM from pure Python/PyTorch line by line, implements multi-head attention, loads pre-trained weights, creates instruction datasets, and implements LoRA from scratch.

### 2.2. Core Deep Learning & NLP Engineering
- **"Deep Learning"** — *Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press)*
  - **Why Read:** The foundational reference for gradient descent, linear algebra, and neural optimizers (like AdamW).
- **"Natural Language Processing with Transformers"** — *Lewis Tunstall, Leandro von Werra, and Thomas Wolf (O'Reilly)*
  - **Why Read:** Written by Hugging Face leads. Explains production workflows with `transformers`, `datasets`, and tokenization pipelines.

---

## 3. High-Value Articles & Interactive Guides

1. **Sebastian Raschka's Blog (Ahead of AI):**
   - Key Post: *"Practical Tips for Finetuning LLMs Using LoRA"* (deep empirical benchmark on optimal $r$, alpha scaling, and target projection layers).
   - [aheadofai.com](https://magazine.aheadofai.com/)
2. **Unsloth Engineering Notes:**
   - Visual guides explaining how manual Triton kernel rewrites achieve 2x–5x training acceleration without accuracy degradation.
   - [docs.unsloth.ai](https://docs.unsloth.ai/)
3. **Jay Alammar — The Illustrated Transformer:**
   - The clearest visual and geometric breakdown of attention matrix interactions.
   - [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
