import re
import sys

import colorama
from colorama import Fore, Style

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

colorama.init(autoreset=True)


def render_markdown_line(line: str, in_code_block: bool) -> tuple[str, bool]:
    """
    Renderiza uma única linha de Markdown para estilos e cores ANSI no terminal.
    Retorna (linha_formatada, novo_estado_in_code_block).
    """
    trimmed = line.strip()

    # Início ou fim de bloco de código ```
    if trimmed.startswith("```"):
        novo_estado = not in_code_block
        tag = trimmed[3:].strip()
        label = f" [código {tag}] " if tag else " [código] "
        return f"{Fore.LIGHTBLACK_EX}---{label}---{Style.RESET_ALL}", novo_estado

    # Dentro de bloco de código: verde claro
    if in_code_block:
        return f"{Fore.LIGHTGREEN_EX}{line}{Style.RESET_ALL}", in_code_block

    # Títulos (#, ##, ###) -> Amarelo brilhante
    m_head = re.match(r"^(#{1,6})\s+(.*)$", line)
    if m_head:
        return (
            f"{Style.BRIGHT}{Fore.YELLOW}{m_head.group(2)}{Style.RESET_ALL}",
            in_code_block,
        )

    # Código inline `code` -> Ciano
    line = re.sub(r"`([^`]+)`", lambda m: f"{Fore.CYAN}{m.group(1)}{Style.RESET_ALL}", line)

    # Negrito **texto** -> Negrito brilhante
    line = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: f"{Style.BRIGHT}{m.group(1)}{Style.RESET_ALL}",
        line,
    )

    # Itálico *texto* -> Limpo
    line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: f"{m.group(1)}", line)

    # Marcadores de lista (- ou *) -> Marcador azul/ciano
    line = re.sub(r"^\s*[-*]\s+", f"  {Fore.CYAN}•{Style.RESET_ALL} ", line)

    # Listas numeradas (1. 2.) -> Número magenta
    line = re.sub(
        r"^\s*(\d+)\.\s+",
        lambda m: f"  {Fore.MAGENTA}{m.group(1)}.{Style.RESET_ALL} ",
        line,
    )

    return line, in_code_block


class LineStreamRenderer:
    """
    Acumula tokens em streaming até formar quebras de linha (`\\n`),
    renderizando cada linha com colorama em tempo real no terminal.
    """

    def __init__(self):
        self.buffer = ""
        self.in_code_block = False

    def feed(self, token: str) -> str:
        self.buffer += token
        if "\n" not in self.buffer:
            return ""

        partes = self.buffer.split("\n")
        # Mantém a última parte (incompleta) no buffer
        self.buffer = partes.pop()

        linhas_renderizadas = []
        for p in partes:
            rend, self.in_code_block = render_markdown_line(p, self.in_code_block)
            linhas_renderizadas.append(rend)

        return "\n".join(linhas_renderizadas) + "\n"

    def flush(self) -> str:
        """Renderiza o restante pendente no buffer."""
        if not self.buffer:
            return ""
        rend, self.in_code_block = render_markdown_line(self.buffer, self.in_code_block)
        self.buffer = ""
        return rend
