"""
Barramento de Mensagens e Estado Efêmero em Tempo Real para a Aigis (Redis Pub/Sub).
Permite sincronização instantânea (<1ms) entre:
- Processo do Discord Bot (interfaces/discord_bot.py)
- Processo de Voz no PC (interfaces/voice_assistant.py)
- Workers Proativos de Agenda/Email (core/proactive_hub.py)
- Futura Interface Live2D (WebSockets/IPC)
"""

import asyncio
import json
import os
from typing import Any, Callable, Coroutine
from loguru import logger
import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Canais canônicos do sistema
CH_DISCORD_EVENTS = "aigis:discord"      # Mensagens, anexos e comandos vindos do Discord
CH_VOICE_EVENTS = "aigis:voice"          # Transcrições de voz, falas e comandos do microfone
CH_PROACTIVE_ALERTS = "aigis:proactive"  # Alertas de agenda, emails, conclusões de ferramentas
CH_PRESENCE_STATE = "aigis:state"        # Estado de presença (falando, ouvindo, ocioso, jogando)


class EventBus:
    _instance: "EventBus | None" = None

    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self._client: aioredis.Redis | None = None
        self._subscriber_tasks: list[asyncio.Task] = []
        self._running = True

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = EventBus()
        return cls._instance

    async def get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                retry_on_timeout=True,
            )
        return self._client

    async def publish(self, channel: str, payload: dict[str, Any]) -> int:
        """Publica um evento JSON em um canal Redis de forma não bloqueante."""
        try:
            client = await self.get_client()
            raw_data = json.dumps(payload, ensure_ascii=False)
            return await client.publish(channel, raw_data)
        except Exception as e:
            logger.warning(f"[EventBus] Falha ao publicar em {channel}: {e}")
            return 0

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None] | None],
    ) -> asyncio.Task:
        """
        Escuta um canal Redis em background e repassa mensagens parseadas para o callback.
        """
        async def _listen_loop():
            sub_client = aioredis.from_url(self.redis_url, decode_responses=True)
            pubsub = sub_client.pubsub()
            await pubsub.subscribe(channel)
            try:
                while self._running:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if msg and msg.get("type") == "message":
                        try:
                            data = json.loads(msg["data"])
                            res = callback(data)
                            if asyncio.iscoroutine(res):
                                await res
                        except Exception as e:
                            logger.error(f"[EventBus] Erro ao processar mensagem em {channel}: {e}")
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                pass
            finally:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.aclose()
                    await sub_client.aclose()
                except Exception:
                    pass

        task = asyncio.create_task(_listen_loop())
        self._subscriber_tasks.append(task)
        return task

    async def set_state(self, key: str, value: Any, expire_seconds: int | None = None) -> bool:
        """Armazena um estado efêmero no Redis."""
        try:
            client = await self.get_client()
            val_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            await client.set(f"state:{key}", val_str, ex=expire_seconds)
            return True
        except Exception as e:
            logger.warning(f"[EventBus] Falha ao salvar estado {key}: {e}")
            return False

    async def get_state(self, key: str) -> Any:
        """Recupera um estado efêmero do Redis."""
        try:
            client = await self.get_client()
            raw = await client.get(f"state:{key}")
            if raw is None:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return raw
        except Exception as e:
            logger.warning(f"[EventBus] Falha ao recuperar estado {key}: {e}")
            return None

    async def close(self):
        """Encerra conexões e listeners do barramento."""
        self._running = False
        for t in self._subscriber_tasks:
            t.cancel()
        if self._client:
            await self._client.aclose()
            self._client = None
