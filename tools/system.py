import json
import os
import re
import subprocess
import urllib.parse
from pathlib import Path

import httpx

# 1. Funções do Sistema

_GLOBAL_PS_SESSION = None


def get_powershell_session():
    """Mantém um processo global persistente do PowerShell para execução ultrarrápida."""
    global _GLOBAL_PS_SESSION
    if _GLOBAL_PS_SESSION is None or _GLOBAL_PS_SESSION.poll() is not None:
        _GLOBAL_PS_SESSION = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            _GLOBAL_PS_SESSION.stdin.write(
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "[Console]::InputEncoding = [System.Text.Encoding]::UTF8\n"
            )
            _GLOBAL_PS_SESSION.stdin.flush()
        except Exception:
            pass
    return _GLOBAL_PS_SESSION


def list_files(path: str = ".") -> str:
    """List files and directories in a given path."""
    try:
        if not path or path.startswith(("/home", "\\home")):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Caminho '{path}' inválido ou inexistente neste sistema (Windows). Se a informação já estiver na memória permanente, use-a em vez de tentar listar diretórios fictícios.",
                },
                ensure_ascii=False,
            )

        items = []
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            t = "Directory" if os.path.isdir(full_path) else "File"
            items.append({"name": item, "type": t})
        return json.dumps({"success": True, "Items": items}, ensure_ascii=False)
    except OSError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"{e}. Lembrete: se estiver procurando projetos do usuário, consulte a memória permanente antes de tentar caminhos no disco.",
            },
            ensure_ascii=False,
        )


def read_file(path: str, max_chars: int = 8000) -> str:
    """Reads the content of a file and returns it as a string with a security limit."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return json.dumps(
                {"success": False, "error": f"File '{path}' does not exist."}, ensure_ascii=False
            )
        if file_path.is_dir():
            return json.dumps(
                {"success": False, "error": f"'{path}' is a directory, not a file."},
                ensure_ascii=False,
            )

        text = file_path.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        content = text[:max_chars]

        return json.dumps(
            {
                "success": True,
                "path": str(file_path),
                "content": content,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def execute_command(command: str, timeout_seconds: int = 15) -> str:
    """
    Executa um comando localmente no terminal do sistema com guardrails de segurança.
    Bloqueia comandos sabidamente destrutivos e impõe timeout de execução.
    """
    # Guardrail contra comandos destrutivos acidentais
    BLOCKED_PATTERNS = [
        r"\bformat\s+[a-zA-Z]:",
        r"\bdel\s+/[sS]\s+",
        r"\brmdir\s+/[sS]\s+",
        r"Remove-Item\s+.*-Recurse\s+.*[cC]:",
        r"\bdrop\s+database\b",
    ]

    for pat in BLOCKED_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Comando '{command}' bloqueado por medidas de segurança de integridade do sistema.",
                },
                ensure_ascii=False,
            )

    # Tenta executar na sessão persistente do PowerShell (~20ms)
    try:
        import uuid

        proc = get_powershell_session()
        marker = str(uuid.uuid4())
        wrapped_cmd = (
            f"$ErrorActionPreference = 'Stop'; "
            f"try {{ {command} }} catch {{ [Console]::Out.WriteLine('__ERR__:' + $_.Exception.Message) }}; "
            f"[Console]::Out.WriteLine('__END_CMD_{marker}__')\n"
        )
        proc.stdin.write(wrapped_cmd)
        proc.stdin.flush()

        stdout_lines = []
        for line in proc.stdout:
            if f"__END_CMD_{marker}__" in line:
                break
            stdout_lines.append(line)

        raw_output = "".join(stdout_lines).strip()
        stderr = ""
        success = True
        returncode = 0

        if "__ERR__:" in raw_output:
            success = False
            returncode = 1
            stderr = raw_output.replace("__ERR__:", "").strip()
            stdout = ""
        else:
            stdout = raw_output

        max_output = 4000
        truncated = len(stdout) > max_output
        if truncated:
            stdout = stdout[:max_output] + "\n...[saída truncada]"

        return json.dumps(
            {
                "success": success,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )
    except Exception:
        # Fallback resiliente para subprocess a frio caso a sessão pipe falhe
        pass

    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )

        stdout = res.stdout.strip()
        stderr = res.stderr.strip()

        # Limite de segurança de saída para não estourar o contexto do LLM
        max_output = 4000
        truncated = len(stdout) > max_output
        if truncated:
            stdout = stdout[:max_output] + "\n...[saída truncada]"

        return json.dumps(
            {
                "success": res.returncode == 0,
                "returncode": res.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "success": False,
                "error": f"Comando excedeu o tempo limite de execução ({timeout_seconds}s).",
            },
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def web_search(query: str, max_results: int = 4) -> str:
    """
    Realiza busca na web em tempo real e retorna títulos, URLs e snippets resumidos.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/", params={"q": query}, headers=headers
            )

            if resp.status_code != 200:
                return json.dumps(
                    {"success": False, "error": f"HTTP {resp.status_code} na busca web."},
                    ensure_ascii=False,
                )

            links = re.findall(
                r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', resp.text, re.DOTALL
            )
            urls = re.findall(r'<a class="result__url"[^>]*href="([^"]+)"', resp.text)

            resultados = []
            for u, s in list(zip(urls, links, strict=False))[:max_results]:
                # Decodifica URL real caso seja o redirect do DuckDuckGo
                real_url = u.strip()
                if "uddg=" in real_url:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(real_url).query)
                    if "uddg" in parsed:
                        real_url = parsed["uddg"][0]
                elif real_url.startswith("//"):
                    real_url = "https:" + real_url

                clean_snippet = re.sub(r"<[^>]+>", "", s).strip()
                resultados.append({"url": real_url, "snippet": clean_snippet})

            if not resultados:
                return json.dumps(
                    {
                        "success": True,
                        "results": [],
                        "message": "Nenhum resultado encontrado para a busca.",
                    },
                    ensure_ascii=False,
                )

            return json.dumps({"success": True, "results": resultados}, ensure_ascii=False)

    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def fetch_url(url: str, max_chars: int = 5000) -> str:
    """
    Baixa o conteúdo textual de uma URL para leitura e síntese pelo assistente.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return json.dumps(
                    {"success": False, "error": f"Erro HTTP {resp.status_code} ao acessar a URL."},
                    ensure_ascii=False,
                )

            html = resp.text
            # Remove scripts e styles
            html = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
            # Remove tags HTML
            text = re.sub(r"<[^>]+>", " ", html)
            # Normaliza espaços
            text = re.sub(r"\s+", " ", text).strip()

            truncated = len(text) > max_chars
            return json.dumps(
                {"success": True, "url": url, "content": text[:max_chars], "truncated": truncated},
                ensure_ascii=False,
            )

    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def remember_facts(facts: list[dict] | dict | None = None, **kwargs) -> str:
    """
    Grava um ou múltiplos fatos permanentes no Grafo e na memória vetorial do PostgreSQL.
    """
    try:
        from core.context import get_local_user
        from memory.embeddings import get_embedding
        from memory.storage import MemoryStorage

        storage = MemoryStorage()
        user = get_local_user()

        lista_fatos = []
        if isinstance(facts, list):
            lista_fatos = facts
        elif isinstance(facts, dict):
            lista_fatos = [facts]
        elif kwargs.get("subject") and kwargs.get("predicate"):
            lista_fatos = [
                {
                    "subject": kwargs["subject"],
                    "predicate": kwargs["predicate"],
                    "object": kwargs.get("object", ""),
                }
            ]

        if not lista_fatos:
            return json.dumps(
                {"success": False, "error": "Nenhum fato válido fornecido."}, ensure_ascii=False
            )

        gravados = []
        for f in lista_fatos:
            subj = str(f.get("subject", "")).strip()
            pred = str(f.get("predicate", "")).strip()
            obj = str(f.get("object", "")).strip()

            if not subj or not pred:
                continue

            storage._sync_add_triplet(subj, pred, obj, user_id=user.user_id)
            frase = f"{subj} {pred} {obj}"
            vec = get_embedding(frase)
            storage._sync_add_memory(
                frase,
                embedding=vec,
                metadata={"source": "remember_facts", "subject": subj},
                user_id=user.user_id,
            )
            gravados.append(f"[{subj}] -> [{pred}] -> [{obj}]")

        return json.dumps(
            {
                "success": True,
                "message": f"{len(gravados)} fatos gravados com sucesso para {user.name}: "
                + ", ".join(gravados),
            },
            ensure_ascii=False,
        )

    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# 2. Schemas JSON para Tool Calling

LIST_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "Lista arquivos e pastas em um diretório existente do disco SOMENTE quando o usuário fornecer um caminho explícito de diretório (ex: 'C:\\Projetos' ou './docs'). NUNCA chame se o usuário não tiver especificado um caminho de diretório.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Caminho explícito de diretório informado pelo usuário.",
                }
            },
            "required": ["path"],
        },
    },
}

READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the content of a file and return it as a string.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the file to be read."}
            },
            "required": ["path"],
        },
    },
}

EXECUTE_COMMAND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Executa um comando no terminal do sistema local (PowerShell/Shell) para verificar status, processos, git, compilar ou rodar scripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Comando exato a ser executado no terminal.",
                }
            },
            "required": ["command"],
        },
    },
}

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Realiza uma pesquisa em tempo real na web para buscar informações recentes, documentações, novidades ou resolver dúvidas técnicas.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Termo de busca a ser pesquisado na internet.",
                }
            },
            "required": ["query"],
        },
    },
}

FETCH_URL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Acessa uma página web a partir de sua URL e extrai o texto para leitura e análise de documentações ou artigos.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Endereço URL completo da página web (ex: 'https://exemplo.com/doc').",
                }
            },
            "required": ["url"],
        },
    },
}

REMEMBER_FACTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "remember_facts",
        "description": "Grava um ou múltiplos fatos permanentes sobre o usuário, preferências ou projetos no banco de memória.",
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "description": "Lista de fatos a serem gravados.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Sujeito (ex: 'Natanael', 'Aigis', 'RLOS')",
                            },
                            "predicate": {
                                "type": "string",
                                "description": "Relação (ex: 'desenvolve', 'usa', 'é_um')",
                            },
                            "object": {
                                "type": "string",
                                "description": "Objeto ou detalhe (ex: 'Rust', 'World Model', 'C++')",
                            },
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                }
            },
            "required": ["facts"],
        },
    },
}

AVAILABLE_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "execute_command": execute_command,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "remember_facts": remember_facts,
    "remember_fact": remember_facts,
}

ALL_TOOL_SCHEMAS = [
    LIST_FILES_SCHEMA,
    READ_FILE_SCHEMA,
    EXECUTE_COMMAND_SCHEMA,
    WEB_SEARCH_SCHEMA,
    FETCH_URL_SCHEMA,
    REMEMBER_FACTS_SCHEMA,
]
