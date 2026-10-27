import asyncio
import os
import re
import sys
import time
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Garante suporte UTF-8 no stdout do Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import discord
from dotenv import load_dotenv

from core.client import chat, load_system_prompt
from core.context import ContextManager, UserContext
from core.timeline import INBOX_DIR, SharedSession
from memory.storage import MemoryStorage

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_OWNER_ID = os.getenv("DISCORD_OWNER_ID", "")
DEDICATED_CHANNEL_NAME = os.getenv("DISCORD_DEDICATED_CHANNEL", "test").lower()

# Tópicos que disparam intervenção proativa do bot em canais gerais
PROACTIVE_TOPICS = [
    r"\brlos\b",
    r"\baigis\b",
    r"\bworld model\b",
    r"\bagente[s]?\b",
    r"\bmem[oó]ria\b",
    r"\barquitetura\b",
    r"\bbootstrap\b",
]

# Cache de sessões por canal/DM para manter contexto recente
sessions: dict[int, ContextManager] = {}
storage = MemoryStorage()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def get_user_context(author: discord.User | discord.Member) -> UserContext:
    """Mapeia usuário do Discord. Identifica dono via DISCORD_OWNER_ID."""
    author_id_str = str(author.id)
    is_owner = bool(DISCORD_OWNER_ID and author_id_str == DISCORD_OWNER_ID.strip())

    return UserContext(
        user_id="local:owner" if is_owner else f"discord:{author_id_str}",
        name=author.display_name,
        is_owner=is_owner,
        platform="discord",
    )


def get_or_create_context(channel_id: int, author: discord.User | discord.Member) -> ContextManager:
    """Recupera ou instancia ContextManager por canal."""
    if channel_id not in sessions:
        user_ctx = get_user_context(author)
        cm = ContextManager(max_chars=25000, user=user_ctx)
        cm.set_system_prompt(load_system_prompt())
        sessions[channel_id] = cm
    else:
        sessions[channel_id].user = get_user_context(author)
    return sessions[channel_id]


def _sync_log_chat(
    channel_id: str,
    channel_name: str,
    author_id: str,
    author_name: str,
    content: str,
    is_bot: bool,
):
    """Grava mensagem no PostgreSQL para futuro dataset de fine-tuning."""
    try:
        with storage._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_logs (channel_id, channel_name, author_id, author_name, content, is_bot)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (channel_id, channel_name, author_id, author_name, content, is_bot),
            )
            conn.commit()
    except Exception:
        pass


async def log_chat(
    channel_id: str,
    channel_name: str,
    author_id: str,
    author_name: str,
    content: str,
    is_bot: bool,
):
    await asyncio.to_thread(
        _sync_log_chat,
        channel_id,
        channel_name,
        author_id,
        author_name,
        content,
        is_bot,
    )


def should_intervene_heuristically(content: str) -> bool:
    """Heurística leve: detecta se a conversa cita projetos-chave ou termos técnicos relevantes."""
    texto = content.lower()
    for pattern in PROACTIVE_TOPICS:
        if re.search(pattern, texto):
            return True
    return False


@client.event
async def on_ready():
    print(f"🤖 Bot conectado no Discord como {client.user} (ID: {client.user.id})")
    try:
        from memory.embeddings import get_embedding

        await asyncio.to_thread(get_embedding, "warmup")
        print("⚡ Warmup de embeddings concluído!")
    except Exception as e:
        print(f"Aviso de warmup: {e}")


@client.event
async def on_message(message: discord.Message):
    # Ignora mensagens geradas pelo próprio bot
    if message.author == client.user:
        return

    canal_nome = getattr(message.channel, "name", "dm").lower()
    canal_id_str = str(message.channel.id)
    autor_id_str = str(message.author.id)

    # 1. Logging passivo contínuo para dataset de fine-tuning
    asyncio.create_task(
        log_chat(
            channel_id=canal_id_str,
            channel_name=canal_nome,
            author_id=autor_id_str,
            author_name=message.author.display_name,
            content=message.clean_content,
            is_bot=False,
        )
    )

    # 2. Critérios de ativação de resposta
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_dedicated = canal_nome == DEDICATED_CHANNEL_NAME
    is_mentioned = client.user in message.mentions
    has_attachments = len(message.attachments) > 0
    is_heuristic = should_intervene_heuristically(message.clean_content)

    # Se não for DM, canal dedicado, menção direta ou trigger heurístico (ou se o dono mandar anexo no canal dedicado/DM), ignora
    if not (is_dm or is_dedicated or is_mentioned or is_heuristic or has_attachments):
        return

    # Processa anexos se houver
    anexos_desc = []
    anexos_meta = []
    if message.attachments:
        for att in message.attachments:
            safe_name = f"{int(time.time())}_{att.filename}"
            caminho_local = INBOX_DIR / safe_name
            try:
                await att.save(str(caminho_local))
                tamanho_kb = att.size / 1024
                snippet = ""
                # Se for arquivo de texto/código ou imagem, adiciona prévia
                if att.content_type and any(
                    t in att.content_type for t in ["text", "json", "python", "javascript", "csv"]
                ):
                    try:
                        conteudo_prev = caminho_local.read_text(encoding="utf-8", errors="replace")[
                            :1000
                        ]
                        snippet = f"\nConteúdo prévio:\n```{conteudo_prev}```"
                    except Exception:
                        pass

                anexos_desc.append(
                    f"- Anexo: {att.filename} ({tamanho_kb:.1f} KB) salvo em `{caminho_local}`{snippet}"
                )
                anexos_meta.append(
                    {
                        "filename": att.filename,
                        "file_path": str(caminho_local),
                        "size": att.size,
                    }
                )
            except Exception as e:
                anexos_desc.append(f"- Anexo {att.filename}: Falha ao salvar ({e})")

    # Limpa menção caso presente
    conteudo = message.clean_content.replace(f"@{client.user.name}", "").strip()
    if anexos_desc:
        conteudo = (conteudo + "\n" if conteudo else "") + "\n".join(anexos_desc)

    if not conteudo:
        return

    user_ctx = get_user_context(message.author)
    shared_session = SharedSession.get_instance()

    # Se for o dono do sistema, integra diretamente na Sessão Unificada (Voz + PC + Discord)
    if user_ctx.is_owner:
        await shared_session.add_user_message(
            source="discord",
            content=conteudo,
            metadata={"attachments": anexos_meta, "channel": canal_nome},
        )
        context = shared_session.context
    else:
        context = get_or_create_context(message.channel.id, message.author)
        context.add_message("user", conteudo)

    async with message.channel.typing():
        resposta_tokens = []
        sent_message = None
        last_edit_time = 0.0
        tools_executed = []

        async for token in chat(
            context.get_messages(),
            user_id=context.user.user_id,
            executed_tools_collector=tools_executed,
        ):
            resposta_tokens.append(token)
            agora = asyncio.get_event_loop().time()

            current_text = "".join(resposta_tokens)
            if not sent_message and current_text.strip():
                sent_message = await message.reply(current_text + " ▌")
                last_edit_time = agora
            elif sent_message and (agora - last_edit_time) > 1.2:
                try:
                    await sent_message.edit(content=current_text + " ▌")
                    last_edit_time = agora
                except Exception:
                    pass

        resposta_completa = "".join(resposta_tokens).strip()
        if not resposta_completa:
            resposta_completa = "Não obtive resposta do modelo."

        context.add_message("assistant", resposta_completa)
        if user_ctx.is_owner:
            meta = {}
            if tools_executed:
                meta = {
                    "tool_calls": [item["tool_call"] for item in tools_executed],
                    "tool_results": [item["result"] for item in tools_executed],
                }
            await shared_session.add_assistant_message(resposta_completa, metadata=meta)
            await shared_session.notify_voice(
                resposta_completa[:200] if len(resposta_completa) > 200 else resposta_completa,
                lang="pt",
            )

        # Grava resposta do bot no dataset
        asyncio.create_task(
            log_chat(
                channel_id=canal_id_str,
                channel_name=canal_nome,
                author_id=str(client.user.id),
                author_name=client.user.name,
                content=resposta_completa,
                is_bot=True,
            )
        )

        if sent_message:
            if len(resposta_completa) <= 2000:
                await sent_message.edit(content=resposta_completa)
            else:
                await sent_message.edit(content=resposta_completa[:1950])
                for chunk in [
                    resposta_completa[i : i + 1950]
                    for i in range(1950, len(resposta_completa), 1950)
                ]:
                    await message.channel.send(chunk)
        else:
            if len(resposta_completa) <= 2000:
                await message.reply(resposta_completa)
            else:
                for chunk in [
                    resposta_completa[i : i + 1950] for i in range(0, len(resposta_completa), 1950)
                ]:
                    await message.channel.send(chunk)


def run_bot():
    if not DISCORD_BOT_TOKEN:
        print("❌ Erro: DISCORD_BOT_TOKEN não configurado no .env")
        return
    client.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    run_bot()
