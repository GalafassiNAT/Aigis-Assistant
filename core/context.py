import getpass
import os
from dataclasses import dataclass


@dataclass
class UserContext:
    """Representa a identidade de quem está interagindo com o agente."""

    user_id: str
    name: str
    is_owner: bool = True
    platform: str = "cli"


def get_local_user() -> UserContext:
    """Descobre dinamicamente o usuário logado no Sistema Operacional."""
    try:
        username = getpass.getuser()
    except Exception:
        username = os.getenv("USERNAME", "local_user")

    return UserContext(user_id=f"local:{username}", name=username, is_owner=True, platform="cli")


class ContextManager:
    def __init__(self, max_chars: int = 30000, user: UserContext | None = None):
        """
        Gerencia o contexto da conversa e o usuário atual da sessão.
        """
        self.max_chars = max_chars
        self.user: UserContext = user or get_local_user()
        self.messages: list[dict] = []

    def set_system_prompt(self, template: str):
        """
        Injeta dinamicamente os dados do usuário atual e o contexto temporal no System Prompt.
        """
        import datetime

        now = datetime.datetime.now()
        data_formatada = now.strftime("%A, %d de %B de %Y (%H:%M)")

        content = template.format(
            username=self.user.name,
            user_id=self.user.user_id,
            user_type="Dono do Sistema (Acesso Completo)"
            if self.user.is_owner
            else "Usuário Convidado (Modo Diálogo)",
            current_date=data_formatada,
        )

        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def add_message(self, role: str, content: str, **kwargs):
        """Adiciona uma mensagem e poda o excesso mantendo o system prompt."""
        msg = {"role": role, "content": content, **kwargs}
        self.messages.append(msg)
        self._prune()

    def get_messages(self) -> list[dict]:
        """Retorna as mensagens prontas para envio à API."""
        return self.messages

    def _total_chars(self) -> int:
        return sum(len(str(m.get("content") or "")) for m in self.messages)

    def _prune(self):
        """
        Remove mensagens antigas mantendo o system prompt (índice 0)
        e preservando a integridade das chamadas de ferramentas.
        """
        while self._total_chars() > self.max_chars and len(self.messages) > 2:
            removed = self.messages.pop(1)
            if (
                removed.get("role") == "assistant"
                and len(self.messages) > 1
                and self.messages[1].get("role") == "tool"
            ):
                self.messages.pop(1)
