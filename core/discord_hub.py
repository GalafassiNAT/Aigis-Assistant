"""
Hub de Gerenciamento de Canais e Comunicados da Aigis no Discord.
Responsável por:
- Assegurar a existência dos canais canônicos dentro da categoria autorizada (ex: 'Botdev').
- Publicar alertas ricos (Embeds) em #aigis-agenda, #aigis-inbox e #aigis-status.
- Marcar o dono (@Nora / Natanael) quando houver notificações prioritárias.
"""

import os
from typing import Any
import discord
from loguru import logger

DISCORD_OWNER_ID = os.getenv("DISCORD_OWNER_ID", "330085110974119939")

# Canais canônicos e suas descrições
CANONICAL_CHANNELS = {
    "aigis-agenda": {
        "topic": "📅 Agenda, prazos acadêmicos, provas e compromissos sincronizados da Aigis.",
        "name_variants": ["aigis-agenda", "📅-aigis-agenda", "agenda"],
    },
    "aigis-inbox": {
        "topic": "📬 Triagem inteligente de emails importantes e notificações externas.",
        "name_variants": ["aigis-inbox", "📬-aigis-inbox", "inbox"],
    },
    "aigis-feed": {
        "topic": "🧠 Feed de contexto rápido: envie prints, links, notas e ideias para a Aigis absorver.",
        "name_variants": ["aigis-feed", "🧠-aigis-feed", "feed-contexto"],
    },
    "aigis-status": {
        "topic": "📊 Status do sistema, conclusão de tarefas em background e logs operacionais.",
        "name_variants": ["aigis-status", "📊-aigis-status", "status-sistema", "reports"],
    },
}


class DiscordHub:
    def __init__(self, client: discord.Client):
        self.client = client
        self.cached_channels: dict[str, discord.TextChannel] = {}

    async def setup_channels(self, guild: discord.Guild, category_name: str = "Botdev") -> dict[str, discord.TextChannel]:
        """
        Verifica ou cria os canais da Aigis dentro da categoria autorizada.
        """
        category = discord.utils.get(guild.categories, name=category_name)
        if not category:
            for cat in guild.categories:
                if cat.name.lower() == category_name.lower():
                    category = cat
                    break

        if not category:
            logger.warning(f"[DiscordHub] Categoria '{category_name}' não encontrada no servidor '{guild.name}'.")
            return {}

        existing_channels = {ch.name.lower(): ch for ch in category.text_channels}

        for key, spec in CANONICAL_CHANNELS.items():
            found_ch = None
            for variant in spec["name_variants"]:
                if variant.lower() in existing_channels:
                    found_ch = existing_channels[variant.lower()]
                    break

            if not found_ch:
                # Canal não existe: tenta criar
                try:
                    target_name = f"aigis-{key.replace('aigis-', '')}"
                    logger.info(f"[DiscordHub] Criando canal #{target_name} na categoria '{category.name}'...")
                    found_ch = await category.create_text_channel(
                        name=target_name,
                        topic=spec["topic"],
                        reason="Estrutura canônica de comunicação da Aigis",
                    )
                    logger.info(f"[DiscordHub] Canal #{target_name} criado com sucesso (ID: {found_ch.id}).")
                except Exception as e:
                    logger.error(f"[DiscordHub] Falha ao criar canal #{key}: {e}")

            if found_ch:
                self.cached_channels[key] = found_ch

        return self.cached_channels

    async def post_agenda_event(
        self,
        title: str,
        time_str: str,
        details: str = "",
        location: str = "",
        ping_owner: bool = False,
    ):
        """Envia card informativo de evento/prova em #aigis-agenda."""
        ch = self.cached_channels.get("aigis-agenda")
        if not ch:
            return

        embed = discord.Embed(
            title=f"📅 {title}",
            description=details or "Compromisso sincronizado no radar da Aigis.",
            color=0x3498DB,
        )
        embed.add_field(name="⏰ Horário / Data", value=time_str, inline=True)
        if location:
            embed.add_field(name="📍 Local / Sala", value=location, inline=True)
        embed.set_footer(text="Aigis Proactive Assistant • Monitor de Rotina")

        content = f"<@{DISCORD_OWNER_ID}>" if ping_owner and DISCORD_OWNER_ID else None
        await ch.send(content=content, embed=embed)

    async def post_inbox_email(
        self,
        sender: str,
        subject: str,
        summary: str,
        date_str: str = "",
        priority: str = "Normal",
        ping_owner: bool = False,
    ):
        """Envia card de triagem de email em #aigis-inbox."""
        ch = self.cached_channels.get("aigis-inbox")
        if not ch:
            return

        color = 0xE74C3C if priority.lower() in ("alta", "urgente", "high") else 0x2ECC71

        embed = discord.Embed(
            title=f"📬 {subject}",
            description=summary,
            color=color,
        )
        embed.add_field(name="De", value=sender, inline=True)
        embed.add_field(name="Prioridade", value=priority.capitalize(), inline=True)
        if date_str:
            embed.add_field(name="Recebido", value=date_str, inline=True)
        embed.set_footer(text="Aigis Inbox Triage • Filtro Inteligente")

        content = f"<@{DISCORD_OWNER_ID}>" if ping_owner and DISCORD_OWNER_ID else None
        await ch.send(content=content, embed=embed)

    async def post_status_log(
        self,
        action: str,
        description: str,
        success: bool = True,
        duration_ms: float | None = None,
        ping_owner: bool = False,
    ):
        """Envia atualização de tarefa ou ferramenta em #aigis-status."""
        ch = self.cached_channels.get("aigis-status")
        if not ch:
            return

        color = 0x2ECC71 if success else 0xE74C3C
        status_icon = "✅" if success else "❌"

        embed = discord.Embed(
            title=f"{status_icon} {action}",
            description=description,
            color=color,
        )
        if duration_ms is not None:
            embed.add_field(name="⏱️ Tempo", value=f"{duration_ms:.1f}ms", inline=True)
        embed.set_footer(text="Aigis System Monitor")

        content = f"<@{DISCORD_OWNER_ID}>" if ping_owner and DISCORD_OWNER_ID else None
        await ch.send(content=content, embed=embed)
