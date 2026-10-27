import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from tools.system import ALL_TOOL_SCHEMAS, AVAILABLE_TOOLS

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("LLAMA_SERVER_URL", "http://localhost:8080/v1"),
    api_key="no_need",
)

MODEL_ALIAS = os.getenv("LLAMA_MODEL_ALIAS", "qwen-coder")


def load_system_prompt() -> str:
    """Carrega as diretrizes do agente a partir do arquivo markdown."""
    return Path("prompts/system.md").read_text(encoding="utf-8")


def should_enable_tools(user_text: str) -> bool:
    """
    Roteador rápido de intenção:
    Se a fala for apenas conversação casual, cumprimento ou alívio (ex: 'oi', 'bom dia', 'boa deu certo'),
    desativa os esquemas de ferramentas para permitir que o modelo responda em ~1 segundo.
    Se contiver intenção de ação (arquivos, comandos, pesquisas, leituras), ativa as ferramentas.
    """
    texto = user_text.lower()
    # Padrões que exigem ferramentas ativas
    ACTION_PATTERNS = [
        r"\b(cria|crie|criar|criando)\b",
        r"\b(apaga|apague|apagar|deleta|delete|deletar)\b",
        r"\b(remove|remova|remover|exclui|exclua|excluir)\b",
        r"\b(l[eê]|leia|ler|mostra|mostre|listar|lista)\b",
        r"\b(pesquisa|pesquise|pesquisar|busca|busque|buscar|procura|procure)\b",
        r"\b(executa|execute|executar|roda|rode|rodar)\b",
        r"\b(arquivo|pasta|diret[oó]rio|vault|nota|script|processo)\b",
        r"\.txt\b|\.md\b|\.py\b|\.json\b",
    ]
    return any(re.search(p, texto) for p in ACTION_PATTERNS)


async def recuperar_contexto_memoria(pergunta: str, user_id: str | None = None) -> str:
    """
    Busca no PostgreSQL (Grafo de Conhecimento + pgvector) fatos e memórias relevantes
    apenas se a pergunta demonstrar necessidade de resgate factual.
    """
    texto = pergunta.lower()
    # Se for apenas saudação ou confirmação, não polui o prompt com tripletos
    if not any(
        k in texto
        for k in (
            "quem",
            "qual",
            "onde",
            "quando",
            "como",
            "lembra",
            "projeto",
            "nick",
            "discord",
            "obsidian",
            "rlos",
            "você sabe",
        )
    ):
        return ""

    try:
        from core.context import get_local_user
        from memory.embeddings import get_embedding
        from memory.storage import MemoryStorage

        storage = MemoryStorage()
        uid = user_id or get_local_user().user_id
        linhas = []
        vistos = set()

        # 1. Fatos do Grafo (Prioridade máxima: fatos atômicos consolidados)
        tripletos = await storage.get_all_triplets(user_id=uid)
        for t in tripletos[:15]:
            fato = f"{t.subject} -> {t.predicate} -> {t.object}"
            if fato not in vistos:
                vistos.add(fato)
                linhas.append(f"- Fato: {fato}")

        # 2. Busca vetorial por similaridade semântica (threshold calibrado para PT-BR: 0.25)
        query_vec = get_embedding(pergunta)
        memorias = await storage.search_memories(
            query_vec, limit=2, min_similarity=0.30, user_id=uid
        )
        for m in memorias:
            if m.content not in vistos:
                vistos.add(m.content)
                linhas.append(f"- Memória: {m.content}")

        if linhas:
            return "\n\n# Contexto Relevante da Memória Permanente:\n" + "\n".join(linhas)
    except Exception:  # noqa: BLE001
        pass
    return ""


def extrair_tool_call(message) -> tuple[str | None, dict | None, str | None]:
    """
    Extrai chamadas de ferramenta suportando:
    1. Formato nativo OpenAI (tool_calls)
    2. Tags XML do Qwen (<function name="..." arguments='...' />)
    3. Blocos de markdown ou tags <tools> / <tool_call>
    4. Qualquer JSON com 'name' e 'arguments' presente no texto
    5. Sintaxe de função Python: func(arg="...")
    """
    # 1. Padrão nativo OpenAI
    if getattr(message, "tool_calls", None):
        tc = message.tool_calls[0]
        try:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            return tc.function.name, args, tc.id
        except Exception:  # noqa: BLE001
            pass

    content = getattr(message, "content", None) or ""
    if not content:
        return None, None, None

    # 2. Formato XML (<function name="..." arguments='...' />)
    m_xml = re.search(r'<function\s+name=["\']([^"\']+)["\']', content)
    if m_xml:
        func_name = m_xml.group(1)
        args_match = re.search(r'arguments=(["\'])(.*?)\1', content, re.DOTALL)
        if args_match:
            try:
                return func_name, json.loads(args_match.group(2)), "call_xml_0"
            except Exception:  # noqa: BLE001
                pass
        return func_name, {}, "call_xml_0"

    # 3. Formato JSON dentro de blocos markdown, <tools> ou <tool_call>
    json_block = re.search(
        r"(?:```(?:json|xml)?|<tool_call>|<tools>)\s*(\{.*?\})\s*(?:```|</tool_call>|</tools>)",
        content,
        re.DOTALL,
    )
    if json_block:
        try:
            dados = json.loads(json_block.group(1))
            if isinstance(dados, dict) and "name" in dados:
                args = dados.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return dados["name"], args, "call_local_0"
        except Exception:  # noqa: BLE001
            pass

    # 4. Qualquer JSON solto contendo 'name'
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            dados = json.loads(content[brace_start : brace_end + 1])
            if isinstance(dados, dict) and "name" in dados:
                args = dados.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return dados["name"], args, "call_local_0"
        except Exception:  # noqa: BLE001
            pass

    # 5. Chamada em sintaxe Python: func(arg="...") ou func(facts=[...])
    for func in AVAILABLE_TOOLS:
        m_func = re.search(rf"\b{func}\s*\((.*?)\)", content, re.DOTALL)
        if m_func:
            raw_args = m_func.group(1).strip()
            args = {}

            # Suporte a listas JSON / Python literals (ex: facts=[{...}])
            m_list = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\[.*?\])", raw_args, re.DOTALL)
            if m_list:
                param_name = m_list.group(1)
                list_str = m_list.group(2)
                try:
                    args[param_name] = json.loads(list_str)
                except Exception:
                    try:
                        import ast

                        args[param_name] = ast.literal_eval(list_str)
                    except Exception:
                        pass

            # Suporte a kwargs normais: chave="valor"
            for m_kw in re.finditer(
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*["\'](.*?)["\']', raw_args, re.DOTALL
            ):
                if m_kw.group(1) not in args:
                    args[m_kw.group(1)] = m_kw.group(2)

            if not args and raw_args:
                m_direct = re.search(r'^["\'](.*?)["\']$', raw_args)
                if m_direct and func in ("read_file", "list_files"):
                    args["path"] = m_direct.group(1)
                elif m_direct and func == "execute_command":
                    args["command"] = m_direct.group(1)
            return func, args, "call_py_0"

    # 6. Fallback de Bloco de Código de Terminal (```powershell ... ``` ou ```bash ... ```)
    m_cmd_block = re.search(
        r"```(?:powershell|pwsh|cmd|bash|sh)?\r?\n(.*?)\r?\n```", content, re.DOTALL
    )
    if m_cmd_block:
        cmd_text = m_cmd_block.group(1).strip()
        # Se contiver comandos típicos de PowerShell ou Shell, executa como execute_command
        if any(
            cmd_text.startswith(k) or f" {k}" in cmd_text
            for k in ("Set-Content", "New-Item", "Get-", "echo", "cat", "dir", "mkdir")
        ):
            return "execute_command", {"command": cmd_text}, "call_block_0"

    return None, None, None


async def chat(
    messages: list[dict],
    max_tool_steps: int = 3,
    user_id: str | None = None,
    on_tool_start: Any = None,
    executed_tools_collector: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Loop de raciocínio (ReAct) com recuperação de memória permanente:
    1. Injeta fatos e memórias relevantes no prompt.
    2. Suporta até max_tool_steps passos de ferramentas encadeadas.
    3. Entrega a resposta final em streaming.
    4. Grava metadados de ferramentas executadas para persistência de contexto.
    """
    # 1. Carrega o System Prompt ESTÁTICO (preserva 100% do cache de prefixo na VRAM do llama-server)
    from core.clock import get_now_ptbr

    now_str = get_now_ptbr()

    payload_messages = [dict(m) for m in messages]
    if not payload_messages or payload_messages[0].get("role") != "system":
        payload_messages.insert(0, {"role": "system", "content": load_system_prompt()})

    # O System Prompt permanece 100% puro e fixo na posição 0 para garantir Prompt Caching na GPU.
    # O relógio biológico exato (hora/minuto) e a memória de longo prazo são anexados
    # no final da mensagem do usuário atual (posição dinâmica).
    if payload_messages and payload_messages[-1].get("role") == "user":
        pergunta = payload_messages[-1].get("content") or ""
        contexto_memoria = await recuperar_contexto_memoria(pergunta, user_id=user_id)

        temporal_info = f"\n[Relógio do Sistema: {now_str}]"
        if contexto_memoria:
            temporal_info += f"\n[Memória Recuperada:\n{contexto_memoria}]"

        payload_messages[-1] = {
            "role": "user",
            "content": f"{pergunta}\n{temporal_info}",
        }

    # Detecta se o turno atual exige esquemas de ferramentas ou se é conversa pura
    user_query = ""
    for m in reversed(payload_messages):
        if m.get("role") == "user":
            user_query = m.get("content", "")
            break

    active_tools = ALL_TOOL_SCHEMAS if should_enable_tools(user_query) else None

    # Se não houver ferramentas ativas, podemos fazer streaming imediato desde o início
    if active_tools is None:
        stream = await client.chat.completions.create(
            model=MODEL_ALIAS,
            messages=payload_messages,
            temperature=0.2,
            max_tokens=1000,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
        return

    # 2. Loop de raciocínio (ReAct) com ferramentas ativas
    for _ in range(max_tool_steps):
        req_kwargs = {
            "model": MODEL_ALIAS,
            "messages": payload_messages,
            "temperature": 0.2,
            "max_tokens": 1000,
            "stream": False,
        }
        if active_tools is not None:
            req_kwargs["tools"] = active_tools

        response = await client.chat.completions.create(**req_kwargs)

        response_message = response.choices[0].message
        func_name, func_args, tool_id = extrair_tool_call(response_message)

        if not func_name or func_name not in AVAILABLE_TOOLS:
            # Resposta direta em texto
            conteudo = response_message.content or ""
            yield conteudo
            return

        # Executa a ferramenta e loga claramente no console para auditoria
        funcao = AVAILABLE_TOOLS[func_name]
        try:
            print(f"\n⚙️ [Executando Ferramenta]: {func_name}({func_args})")
        except Exception:
            print(f"\n[Executando Ferramenta]: {func_name}({func_args})")

        # Dispara feedback imediato (ex: áudio "Processando...") sem bloquear
        if on_tool_start is not None:
            try:
                res_cb = on_tool_start(func_name, func_args)
                if asyncio.iscoroutine(res_cb):
                    asyncio.create_task(res_cb)
            except Exception:
                pass

        try:
            resultado = funcao(**func_args)
            try:
                print(f"   Resultado: {resultado}")
            except Exception:
                print("   Resultado obtido.")
        except Exception as e:  # noqa: BLE001
            resultado = json.dumps({"success": False, "error": str(e)})
            try:
                print(f"   Erro na execução: {e}")
            except Exception:
                print("   Erro na execucao da ferramenta.")

        call_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id or "call_0",
                    "type": "function",
                    "function": {"name": func_name, "arguments": json.dumps(func_args)},
                }
            ],
        }
        messages.append(call_msg)
        payload_messages.append(call_msg)

        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_id or "call_0",
            "content": str(resultado),
        }
        messages.append(tool_msg)
        payload_messages.append(tool_msg)

        if executed_tools_collector is not None:
            executed_tools_collector.append(
                {
                    "tool_call": {
                        "id": tool_id or "call_0",
                        "type": "function",
                        "function": {"name": func_name, "arguments": json.dumps(func_args)},
                    },
                    "result": {
                        "tool_call_id": tool_id or "call_0",
                        "content": str(resultado),
                    },
                }
            )

    # 3. Gera resposta final em stream após executar as ferramentas
    stream = await client.chat.completions.create(
        model=MODEL_ALIAS,
        messages=payload_messages,
        temperature=0.2,
        max_tokens=1000,
        stream=True,
    )

    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token
