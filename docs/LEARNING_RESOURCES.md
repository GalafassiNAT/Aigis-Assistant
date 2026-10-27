# Project AIGIS — Guia de Estudos, Artigos & Leituras Recomendadas

Este documento reúne os principais **artigos científicos seminais (*papers*)**, **livros de referência** e **conteúdos práticos** para aprofundar o entendimento sobre as tecnologias utilizadas no projeto: arquitetura de Transformers, técnicas de adaptação eficiente (LoRA/PEFT), quantização, *tool use* e engenharia de sistemas com LLMs locais.

---

## 1. Artigos Seminais Fundamentais (*Research Papers*)

### 1.1. Adaptação Eficiente & Fine-Tuning (PEFT / LoRA / QLoRA)
1. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
   - *O que explica:* O paper original que introduziu a técnica de decomposição matricial $W + B \times A$. Leitura obrigatória para entender a matemática por trás da redução de 99% dos gradientes treináveis.
   - *Link:* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
2. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
   - *O que explica:* Introduziu o tipo de dado `NF4` (NormalFloat4), *Double Quantization* e paginação de gradientes, permitindo treinar modelos de 7B–65B em GPUs de consumo de 8GB–16GB (como a T4 do Colab).
   - *Link:* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
3. **Instruction Tuning with GPT-4** (Peng et al., 2023)
   - *O que explica:* Como datasets sintéticos multi-turn refinam a aderência a instruções e a persona do assistente.
   - *Link:* [arXiv:2304.03277](https://arxiv.org/abs/2304.03277)

### 1.2. Arquitetura de Transformers & Modelos Base
4. **Attention Is All You Need** (Vaswani et al., 2017)
   - *O que explica:* O artigo fundamental que inventou o Transformer, explicando o mecanismo de *Scaled Dot-Product Attention*, projeções Q, K, V e matrizes multi-head.
   - *Link:* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
5. **Qwen2.5 Technical Report** (Qwen Team, 2024)
   - *O que explica:* A arquitetura específica do modelo que usamos como cérebro da Aigis (Qwen2.5-Coder), incluindo RoPE, SwiGLU e janelas de contexto estendidas.
   - *Link:* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

### 1.3. Agentes, Raciocínio & Execução de Ferramentas (*Tool Use*)
6. **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022)
   - *O que explica:* O padrão arquitetural que usamos no `core/client.py` (intercalar Pensamento -> Ação/Ferramenta -> Observação -> Resposta final).
   - *Link:* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
7. **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023)
   - *O que explica:* Como modelos aprendem autonomamente a decidir *quando* e *quais* APIs chamar para resolver tarefas.
   - *Link:* [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)

---

## 2. Livros Recomendados

### 2.1. O Livro Definitivo para o seu Momento
- **"Build a Large Language Model (from Scratch)"** — *Sebastian Raschka (Manning Publications, 2024)*
  - **Por que ler:** É sem dúvida o melhor livro da atualidade para quem quer entender tudo por dentro. Ele constrói um LLM completo linha por linha em Python/PyTorch puro, implementa o mecanismo de atenção, pré-treina o modelo e depois ensina exatamente a criar datasets de instrução e implementar LoRA do zero.

### 2.2. Fundamentos de Deep Learning & Engenharia de IA
- **"Deep Learning"** — *Ian Goodfellow, Yoshua Bengio e Aaron Courville (MIT Press)*
  - **Por que ler:** A bíblia matemática de redes neurais, álgebra linear, gradientes e otimizadores (como o AdamW que configuramos no script).
- **"Natural Language Processing with Transformers"** — *Lewis Tunstall, Leandro von Werra e Thomas Wolf (O'Reilly)*
  - **Por que ler:** Escrito pelos próprios criadores e engenheiros chefes da Hugging Face. Ensina a usar a biblioteca `transformers`, `datasets` e `accelerate` na prática.

---

## 3. Blogs, Repositórios e Guias Práticos na Web

1. **Blog do Sebastian Raschka (Ahead of AI):**
   - Artigo seminal: *"Practical Tips for Finetuning LLMs Using LoRA"* (analisa benchmarks empíricos sobre qual valor de $r$ escolher, quais camadas adaptar e qual taxa de aprendizado usar).
   - [aheadofai.com](https://magazine.aheadofai.com/)
2. **Unsloth Documentation & Notebooks:**
   - Guias visuais e explicações técnicas de como eles aceleram o treino em 2x–5x reescrevendo kernels CUDA em Triton.
   - [docs.unsloth.ai](https://docs.unsloth.ai/)
3. **Jay Alammar — The Illustrated Transformer:**
   - O melhor guia visual intuitivo sobre como vetores e pesos viajam dentro das matrizes de atenção.
   - [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
