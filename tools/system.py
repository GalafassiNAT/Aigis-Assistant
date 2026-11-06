import concurrent.futures
import json
import os
import re
import subprocess
import sys
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
        target_path = Path(path or ".").resolve()
        if not target_path.exists():
            return json.dumps(
                {
                    "success": False,
                    "error": f"Caminho '{path}' não existe.",
                },
                ensure_ascii=False,
            )

        items = []
        for item in os.listdir(target_path):
            full_path = target_path / item
            t = "Directory" if full_path.is_dir() else "File"
            items.append({"name": item, "type": t})
        return json.dumps({"success": True, "Items": items}, ensure_ascii=False)
    except OSError as e:
        return json.dumps(
            {
                "success": False,
                "error": str(e),
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
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f*|-rf|-fr)\s+/\b",
        r"\bmkfs\b",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
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

    if sys.platform != "win32":
        try:
            res = subprocess.run(
                ["/bin/bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
            stdout = res.stdout.strip()
            stderr = res.stderr.strip()
            max_output = 1200
            truncated = len(stdout) > max_output
            if truncated:
                stdout = stdout[:max_output] + "\n...[saída truncada para síntese de voz e economia de contexto]"
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

    # Windows: Tenta executar na sessão persistente do PowerShell (~20ms)
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

        def _read_stdout():
            stdout_lines = []
            for line in proc.stdout:
                if f"__END_CMD_{marker}__" in line:
                    break
                stdout_lines.append(line)
            return "".join(stdout_lines).strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_read_stdout)
            try:
                raw_output = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                # Mata a sessão travada para evitar vazamento de processo e força recriação limpa
                try:
                    proc.kill()
                except Exception:
                    pass
                global _GLOBAL_PS_SESSION
                _GLOBAL_PS_SESSION = None
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Comando excedeu o tempo limite de execução ({timeout_seconds}s) na sessão persistente do PowerShell.",
                    },
                    ensure_ascii=False,
                )

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

        max_output = 1200
        truncated = len(stdout) > max_output
        if truncated:
            stdout = stdout[:max_output] + "\n...[saída truncada para síntese de voz e economia de contexto]"

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


def create_file(path: str, content: str = "") -> str:
    """Cria um arquivo no disco e verifica sua existência real."""
    try:
        p = Path(path).expanduser().resolve()
        blocked_prefixes = ["/etc", "/bin", "/sbin", "/usr", "/lib", "/boot", "/sys", "/dev", "C:\\Windows"]
        if any(str(p).startswith(b) for b in blocked_prefixes):
            return json.dumps({"success": False, "error": f"Acesso negado para criar arquivo em '{p}'."})

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        verified = p.exists() and p.is_file()
        return json.dumps(
            {
                "success": verified,
                "path": str(p),
                "verified": verified,
                "message": f"Arquivo '{p.name}' criado com sucesso e verificado em disco."
                if verified
                else "Falha ao verificar arquivo no disco.",
            },
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def delete_file(path: str) -> str:
    """Remove um arquivo do disco e verifica sua exclusão real."""
    try:
        p = Path(path).expanduser().resolve()
        blocked_prefixes = ["/etc", "/bin", "/sbin", "/usr", "/lib", "/boot", "/sys", "/dev", "C:\\Windows"]
        if any(str(p).startswith(b) for b in blocked_prefixes):
            return json.dumps({"success": False, "error": f"Acesso negado para remover '{p}'."})

        if not p.exists():
            return json.dumps(
                {"success": False, "error": f"Arquivo '{p.name}' não encontrado no caminho especificado."},
                ensure_ascii=False,
            )

        if p.is_dir():
            return json.dumps(
                {"success": False, "error": f"'{p.name}' é um diretório, não um arquivo."},
                ensure_ascii=False,
            )

        p.unlink()
        verified = not p.exists()
        return json.dumps(
            {
                "success": verified,
                "path": str(p),
                "verified": verified,
                "message": f"Arquivo '{p.name}' removido com sucesso e verificado no sistema."
                if verified
                else "Falha ao remover arquivo do disco.",
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


def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str | None = None,
    date: str | None = None,
    description: str = "",
    location: str = "",
) -> str:
    """
    Cria um novo evento/apontamento na agenda do Google Calendar do usuário.
    date: Data no formato 'YYYY-MM-DD' (ex: '2026-11-09') ou nome do dia ('segunda', 'terça', etc.).
    start_time: Horário de início ('HH:MM' como '22:30' ou ISO completo).
    """
    try:
        from datetime import datetime, timedelta
        from pathlib import Path
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        token_path = Path("token.json")
        if not token_path.exists():
            return json.dumps(
                {"success": False, "error": "Credenciais do Google Calendar não encontradas (token.json ausente)."},
                ensure_ascii=False,
            )

        creds = Credentials.from_authorized_user_file(str(token_path))
        service = build("calendar", "v3", credentials=creds)

        now = datetime.now()
        target_date = now.date()

        # Resolução inteligente de dia da semana se fornecido
        DIAS_MAP = {
            "segunda": 0, "segunda-feira": 0,
            "terca": 1, "terça": 1, "terca-feira": 1, "terça-feira": 1,
            "quarta": 2, "quarta-feira": 2,
            "quinta": 3, "quinta-feira": 3,
            "sexta": 4, "sexta-feira": 4,
            "sabado": 5, "sábado": 5,
            "domingo": 6,
        }

        if date:
            clean_date = date.strip().lower()
            if clean_date in ("amanhã", "amanha"):
                target_date = (now + timedelta(days=1)).date()
            elif clean_date in ("hoje",):
                target_date = now.date()
            elif clean_date in DIAS_MAP:
                target_weekday = DIAS_MAP[clean_date]
                days_ahead = (target_weekday - now.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_date = (now + timedelta(days=days_ahead)).date()
            else:
                try:
                    target_date = datetime.strptime(clean_date, "%Y-%m-%d").date()
                except Exception:
                    pass

        # Monta o ISO com base no target_date
        if "T" in start_time:
            start_iso = start_time
        else:
            m = re.match(r"^(\d{1,2}):(\d{2})$", start_time.strip())
            if m:
                hh, mm = int(m.group(1)), int(m.group(2))
                start_dt = datetime(target_date.year, target_date.month, target_date.day, hh, mm)
                start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:00-03:00")
            else:
                start_iso = f"{target_date.isoformat()}T{start_time}:00-03:00"

        if not end_time or "T" not in end_time:
            try:
                start_dt_obj = datetime.fromisoformat(start_iso)
                end_iso = (start_dt_obj + timedelta(hours=1)).isoformat()
            except Exception:
                end_iso = start_iso
        else:
            end_iso = end_time

        event_body = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"dateTime": start_iso, "timeZone": "America/Sao_Paulo"},
            "end": {"dateTime": end_iso, "timeZone": "America/Sao_Paulo"},
        }

        created = service.events().insert(calendarId="primary", body=event_body).execute()
        html_link = created.get("htmlLink", "")

        return json.dumps(
            {
                "success": True,
                "event_id": created.get("id"),
                "title": title,
                "start": start_iso,
                "end": end_iso,
                "link": html_link,
                "message": f"Evento '{title}' criado com sucesso no Google Calendar para {start_iso}.",
            },
            ensure_ascii=False,
        )

    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": f"Erro ao criar evento: {e}"}, ensure_ascii=False)


def delete_calendar_event(event_id: str | None = None, title: str | None = None) -> str:
    """
    Remove ou apaga um evento do Google Calendar por ID ou por busca de título recente.
    """
    try:
        from pathlib import Path
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        token_path = Path("token.json")
        if not token_path.exists():
            return json.dumps({"success": False, "error": "token.json ausente."}, ensure_ascii=False)

        creds = Credentials.from_authorized_user_file(str(token_path))
        service = build("calendar", "v3", credentials=creds)

        target_id = event_id
        if not target_id and title:
            # Busca o evento pelo título
            events = service.events().list(calendarId="primary", q=title, maxResults=5).execute().get("items", [])
            if events:
                target_id = events[0]["id"]
            else:
                return json.dumps({"success": False, "error": f"Nenhum evento encontrado com título '{title}'."}, ensure_ascii=False)

        if not target_id:
            return json.dumps({"success": False, "error": "Informe event_id ou title para exclusão."}, ensure_ascii=False)

        service.events().delete(calendarId="primary", eventId=target_id).execute()
        return json.dumps({"success": True, "message": f"Evento '{target_id}' removido com sucesso da agenda."}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": f"Erro ao excluir evento: {e}"}, ensure_ascii=False)


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
        "description": "Executa um comando no terminal Bash do Linux para criar, modificar ou apagar arquivos e pastas (ex: touch, rm -f, mkdir), rodar scripts, gerenciar processos ou checar o sistema.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Comando exato a ser executado no terminal (ex: 'touch /home/nora/Desktop/teste.txt' ou 'rm -f ...').",
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

CREATE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Cria um novo arquivo no disco com caminho e conteúdo especificados e verifica sua existência real.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Caminho absoluto do arquivo a ser criado (ex: '/home/nora/Desktop/teste.txt').",
                },
                "content": {
                    "type": "string",
                    "description": "Conteúdo inicial do arquivo de texto (opcional).",
                },
            },
            "required": ["path"],
        },
    },
}

DELETE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "Remove ou apaga um arquivo existente no disco com verificação pós-exclusão.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Caminho absoluto do arquivo a ser removido (ex: '/home/nora/Desktop/teste.txt').",
                }
            },
            "required": ["path"],
        },
    },
}

CREATE_CALENDAR_EVENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Cria um novo evento, reunião ou compromisso diretamente na agenda do Google Calendar do usuário.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Título ou resumo do evento (ex: 'Reunião de Alinhamento', 'Apontamento de Teste', 'Dentista').",
                },
                "start_time": {
                    "type": "string",
                    "description": "Horário de início ('HH:MM' como '22:30' ou ISO completo).",
                },
                "date": {
                    "type": "string",
                    "description": "Data do evento ('YYYY-MM-DD' como '2026-11-09', ou dia da semana como 'segunda', 'terça', 'hoje', 'amanhã').",
                },
                "end_time": {
                    "type": "string",
                    "description": "Horário de término (opcional, padrão: 1 hora após o início).",
                },
                "description": {
                    "type": "string",
                    "description": "Descrição detalhada do evento (opcional).",
                },
                "location": {
                    "type": "string",
                    "description": "Local físico ou link da reunião (opcional).",
                },
            },
            "required": ["title", "start_time"],
        },
    },
}

DELETE_CALENDAR_EVENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_calendar_event",
        "description": "Exclui um evento da agenda do Google Calendar por título ou ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Título do evento a ser excluído (ex: 'Apontamento de Teste').",
                },
                "event_id": {
                    "type": "string",
                    "description": "ID exato do evento (opcional).",
                },
            },
        },
    },
}

AVAILABLE_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "create_file": create_file,
    "delete_file": delete_file,
    "execute_command": execute_command,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "remember_facts": remember_facts,
    "remember_fact": remember_facts,
    "create_calendar_event": create_calendar_event,
    "delete_calendar_event": delete_calendar_event,
}

ALL_TOOL_SCHEMAS = [
    LIST_FILES_SCHEMA,
    READ_FILE_SCHEMA,
    CREATE_FILE_SCHEMA,
    DELETE_FILE_SCHEMA,
    EXECUTE_COMMAND_SCHEMA,
    WEB_SEARCH_SCHEMA,
    FETCH_URL_SCHEMA,
    REMEMBER_FACTS_SCHEMA,
    CREATE_CALENDAR_EVENT_SCHEMA,
    DELETE_CALENDAR_EVENT_SCHEMA,
]
