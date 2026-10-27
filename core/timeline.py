import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.client import load_system_prompt
from core.context import ContextManager, get_local_user
from memory.storage import MemoryStorage

INBOX_DIR = Path("data/inbox").resolve()
INBOX_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TimelineEvent:
    source: str
    author: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SharedSession:
    """
    Timeline unificada da Aigis sincronizada via PostgreSQL.
    Permite que o processo do Discord e o processo de Voz no PC
    compartilhem exatamente a mesma memória em tempo real.
    """

    _instance: "SharedSession | None" = None

    def __init__(self):
        self.user = get_local_user()
        self.storage = MemoryStorage()
        self.context = ContextManager(max_chars=30000, user=self.user)
        self.context.set_system_prompt(load_system_prompt())
        self._lock = asyncio.Lock()
        self._on_voice_notify: Any = None

    @classmethod
    def get_instance(cls) -> "SharedSession":
        if cls._instance is None:
            cls._instance = SharedSession()
        return cls._instance

    def register_voice_notifier(self, callback):
        self._on_voice_notify = callback

    async def notify_voice(self, text: str):
        if self._on_voice_notify:
            try:
                if asyncio.iscoroutinefunction(self._on_voice_notify):
                    await self._on_voice_notify(text)
                else:
                    self._on_voice_notify(text)
            except Exception as e:
                print(f"[Timeline] Erro ao emitir notificação de voz: {e}")

    def _sync_add_event(self, source: str, author: str, content: str, metadata: dict[str, Any]):
        try:
            with self.storage._get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO shared_timeline (source, author, content, metadata)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (source, author, content, json.dumps(metadata)),
                )
                conn.commit()
        except Exception as e:
            print(f"[Timeline DB] Erro ao salvar evento: {e}")

    async def add_user_message(
        self, source: str, content: str, metadata: dict[str, Any] | None = None
    ):
        meta = metadata or {}
        # 1. Grava no banco compartilhado
        await asyncio.to_thread(self._sync_add_event, source, self.user.name, content, meta)
        # 2. Mantém no ContextManager local
        prefix = (
            "[Voz no PC]"
            if source == "voice"
            else "[Discord]"
            if source == "discord"
            else f"[{source}]"
        )
        self.context.add_message("user", f"{prefix}: {content}")

    async def add_assistant_message(self, content: str, metadata: dict[str, Any] | None = None):
        meta = metadata or {}
        await asyncio.to_thread(self._sync_add_event, "assistant", "Aigis", content, meta)
        self.context.add_message("assistant", content)

    def _sync_get_recent_events(self, limit: int = 15) -> list[dict]:
        try:
            with self.storage._get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source, author, content, metadata, created_at
                    FROM shared_timeline
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                # Ordena cronologicamente
                rows.reverse()
                return [
                    {
                        "source": r[0],
                        "author": r[1],
                        "content": r[2],
                        "metadata": r[3] or {},
                        "created_at": r[4],
                    }
                    for r in rows
                ]
        except Exception:
            return []

    async def get_synchronized_messages(self) -> list[dict]:
        """
        Reconstrói o contexto misturando os últimos eventos do PostgreSQL
        (garante que voz veja o discord e discord veja a voz instantaneamente).
        Reconstitui tool_calls e tool results gravados no metadata para que
        o modelo de raciocínio compreenda o fluxo real de ações executadas.
        """
        events = await asyncio.to_thread(self._sync_get_recent_events, 20)
        if not events:
            return self.context.get_messages()

        msgs = []
        # Mantém o system prompt
        if self.context.messages and self.context.messages[0].get("role") == "system":
            msgs.append(self.context.messages[0])

        for ev in events:
            src = ev["source"]
            content = ev["content"]
            meta = ev.get("metadata") or {}

            if src == "assistant":
                # Se esse turno teve chamadas de ferramentas, reconstrói o histórico OpenAPI
                tool_calls = meta.get("tool_calls")
                tool_results = meta.get("tool_results")
                if tool_calls and tool_results:
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }
                    )
                    for res in tool_results:
                        msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": res.get("tool_call_id", "call_0"),
                                "content": res.get("content", ""),
                            }
                        )
                msgs.append({"role": "assistant", "content": content})
            else:
                prefix = "[Comando por Voz no PC]" if src == "voice" else "[Mensagem via Discord]"
                msgs.append({"role": "user", "content": f"{prefix}: {content}"})

        return msgs

    def get_messages(self) -> list[dict]:
        return self.context.get_messages()
