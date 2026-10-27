# Especificação de Comportamento e Dinâmica do Agente ("Core Persona & Loop")

## 1. O Propósito e a Experiência do Usuário (UX)
* **Não é um Chatbot Passivo:** O agente não deve apenas aguardar comandos estáticos. Ele tem autonomia para analisar o contexto do ambiente (código aberto, logs de erro, comandos executados) e sugerir intervenções ou tirar dúvidas de forma proativa.
* **Personalidade e Tom:** Um assistente técnico, afiado, direto, sem formalidades robóticas excessivas, focado em resolver problemas de arquitetura de software, automação e desenvolvimento de sistemas. Respostas conversacionais e fluidas (evitando blocos gigantes de texto quando o foco for diálogo rápido).

## 2. O Ciclo de Vida da Informação (Loop Cognitivo)
Para evitar amnésia e garantir coerência de longo prazo, o agente opera em um ciclo contínuo:
1. **Percepção (Input):** Recebe o estímulo via Discord (texto/código) ou eventos locais do sistema.
2. **Consulta de Memória (Graph RAG + Vetores):** Antes de responder, o orquestrador consulta o grafo de conhecimento no SSD para resgatar quem é o usuário, quais projetos estão ativos (ex: arquitetura do Aigis, configurações do Arch Linux, scripts em Rust/TypeScript) e preferências passadas.
3. **Processamento (Cérebro Local):** O `llama-server` processa a inferência na RX 7600 a ~40 t/s utilizando uma janela de contexto otimizada (`ctx-size = 16384` com cache quantizado `q8_0`).
4. **Ação / Resposta (Tool Use / Output):** O agente decide se apenas responde no chat, se executa uma ferramenta no sistema operacional (via `subprocess`/Python) ou se atualiza o banco de dados de grafo com novos fatos aprendidos durante a conversa.

## 3. Gestão de Memória e Aprendizado Contínuo
* **Fatos vs. Conversas:** Mensagens corriqueiras de chat expiram na janela de curto prazo. No entanto, decisões técnicas, preferências de ferramentas, estrutura de projetos e correções de bugs são destiladas por um prompt secundário de fundo e inseridas permanentemente no **Grafo de Conhecimento**.
* **Isolamento de Contexto por Tópico:** O agente sabe diferenciar quando o usuário está conversando sobre programação de baixo nível (Rust/Zig) versus automação de sistema ou conversas gerais, ajustando o foco do prompt dinamicamente.

## 4. Integração Prática com o Ambiente do Usuário
* **Canal Principal:** Servidor privado ou DMs dedicadas no Discord para acesso rápido de qualquer lugar.
* **Ferramentas Nativas Disponíveis para o Agente:**
  * Leitura e análise de arquivos de código locais no workspace.
  * Execução controlada de diagnósticos de sistema e comandos de terminal.
  * Varredura na base de conhecimento estruturada (grafo) para lembrar de decisões arquiteturais antigas.