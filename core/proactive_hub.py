"""
Motor de Vida e Proatividade da Aigis (Estilo Toki / Companheira de Vida).
Responsável por:
1. Varrer o Obsidian Vault (Atividades e Provas, Daily Notes, Calendários) para compilar sua rotina.
2. Integrar com Google Calendar e Gmail para capturar compromissos e triagem inteligente de emails.
3. Notificar em tempo real através do barramento Redis (EventBus) e postar cards ricos no DiscordHub.
4. Alimentar o contexto de voz para saudações matinais e conversação orgânica.
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger

from core.bus import CH_PROACTIVE_ALERTS, EventBus

VAULT_DIR = Path(
    os.getenv(
        "OBSIDIAN_VAULT_PATH",
        "/home/nora/Obsidian Vaults/Vault-Obisidian",
    )
)


NOTIFIED_EMAILS_FILE = Path("data/notified_emails.json")

AUTOMATED_SENDER_PATTERNS = [
    r"no[-_]?reply",
    r"donotreply",
    r"do-not-reply",
    r"nao[-_]?responda",
    r"notification",
    r"notificac",
    r"newsletter",
    r"news@",
    r"boletim",
    r"digest",
    r"updates@",
    r"replies@",
    r"jobalerts",
    r"jobs-noreply",
    r"mailer[-_]?daemon",
    r"googlecommunityteam",
    r"googledev",
    r"accounts\.google\.com",
    r"mailer\.udemy",
    r"mail\.c6bank",
    r"oracle-mail",
    r"uber\.com",
    r"linkedin\.com",
    r"bounce",
]


class ProactiveLifeHub:
    def __init__(self, vault_path: Path = VAULT_DIR):
        self.vault_path = vault_path
        self.bus = EventBus.get_instance()

    def _load_notified_emails(self) -> set[str]:
        if NOTIFIED_EMAILS_FILE.exists():
            try:
                data = json.loads(NOTIFIED_EMAILS_FILE.read_text(encoding="utf-8"))
                return set(data)
            except Exception:
                pass
        return set()

    def _save_notified_email(self, email_id: str):
        try:
            current = self._load_notified_emails()
            current.add(email_id)
            NOTIFIED_EMAILS_FILE.parent.mkdir(parents=True, exist_ok=True)
            NOTIFIED_EMAILS_FILE.write_text(json.dumps(list(current)), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[ProactiveHub] Erro ao salvar notified_email: {e}")

    # =========================================================================
    # 1. Scanner Local do Obsidian Vault
    # =========================================================================

    def scan_obsidian_tasks_and_exams(self) -> list[dict[str, Any]]:
        """Extrai tarefas pendentes e provas do Kanban e notas do Vault."""
        items: list[dict[str, Any]] = []
        if not self.vault_path.exists():
            return items

        # A) Atividades e Provas.md (Kanban)
        atividades_file = self.vault_path / "Atividades e Provas.md"
        if atividades_file.exists():
            try:
                content = atividades_file.read_text(encoding="utf-8")
                matches = re.finditer(
                    r"- \[([ xX])\] (?:\[\[(.*?)\]\]|(.*?))(?:\n|\Z)(.*?)(?=\n- \[|\Z|%%)",
                    content,
                    re.DOTALL,
                )
                for m in matches:
                    done = m.group(1).lower() == "x"
                    title = (m.group(2) or m.group(3) or "").strip()
                    extra = (m.group(4) or "").strip()

                    # Procura data e sala nos detalhes
                    date_m = re.search(r"Data:\s*([^\n\r]+)", extra, re.IGNORECASE)
                    sala_m = re.search(r"Sala:\s*([^\n\r]+)", extra, re.IGNORECASE)

                    date_str = date_m.group(1).strip() if date_m else ""
                    sala_str = sala_m.group(1).strip() if sala_m else ""

                    if title:
                        items.append(
                            {
                                "source": "obsidian_kanban",
                                "title": title,
                                "done": done,
                                "date": date_str,
                                "location": sala_str,
                                "raw_details": extra,
                            }
                        )
            except Exception as e:
                logger.warning(f"[ProactiveHub] Erro ao ler Atividades e Provas: {e}")

        # B) Calendário do College
        cal_dir = self.vault_path / "College" / "Calendar"
        if cal_dir.exists():
            try:
                for f in sorted(cal_dir.glob("*.md")):
                    # Padrão: YYYY-MM-DD Disciplina - Prova X.md
                    m_date = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(.*)$", f.stem)
                    if m_date:
                        date_str = m_date.group(1)
                        desc = m_date.group(2)
                        items.append(
                            {
                                "source": "obsidian_college_calendar",
                                "title": desc,
                                "done": False,
                                "date": date_str,
                                "location": "Faculdade",
                                "raw_details": f"Registro no calendário acadêmico: {f.name}",
                            }
                        )
            except Exception as e:
                logger.warning(f"[ProactiveHub] Erro ao escanear calendário do College: {e}")

        return items

    def get_upcoming_briefing(self) -> dict[str, Any]:
        """Compila um resumo estruturado para o briefing matinal e conversa da Aigis."""
        all_items = self.scan_obsidian_tasks_and_exams()
        pending_items = [it for it in all_items if not it.get("done")]

        briefing_text_lines = []
        for it in pending_items[:5]:
            line = f"- {it['title']}"
            if it.get("date"):
                line += f" (Data: {it['date']})"
            if it.get("location"):
                line += f" [Local: {it['location']}]"
            briefing_text_lines.append(line)

        briefing_summary = "\n".join(briefing_text_lines) if briefing_text_lines else "Nenhum compromisso ou prova pendente no momento."

        return {
            "total_pending": len(pending_items),
            "items": pending_items,
            "summary_text": briefing_summary,
            "generated_at": datetime.now().isoformat(),
        }

    # =========================================================================
    # 2. Conector Google Calendar & Gmail (Stubs prontos para OAuth2)
    # =========================================================================

    def is_google_configured(self) -> bool:
        """Verifica se existem credenciais do Google configuradas localmente."""
        cred_file = Path("credentials.json")
        token_file = Path("token.json")
        return cred_file.exists() or token_file.exists()

    async def fetch_google_calendar_events(self, max_results: int = 5) -> list[dict[str, Any]]:
        """
        Consulta os próximos eventos do Google Calendar se as credenciais existirem.
        Caso contrário, retorna lista vazia de forma graciosa.
        """
        if not self.is_google_configured():
            return []

        # Stub de execução OAuth2 seguro
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            token_path = Path("token.json")
            if not token_path.exists():
                return []

            creds = Credentials.from_authorized_user_file(str(token_path))
            service = build("calendar", "v3", credentials=creds)

            now = datetime.utcnow().isoformat() + "Z"
            events_result = await asyncio.to_thread(
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute
            )
            events = events_result.get("items", [])
            parsed = []
            for ev in events:
                start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date"))
                parsed.append(
                    {
                        "source": "google_calendar",
                        "title": ev.get("summary", "Sem título"),
                        "date": start,
                        "location": ev.get("location", ""),
                        "details": ev.get("description", ""),
                    }
                )
            return parsed
        except Exception as e:
            logger.warning(f"[ProactiveHub] Google Calendar indisponível: {e}")
            return []

    async def fetch_unread_gmails(self, max_results: int = 5) -> list[dict[str, Any]]:
        """
        Consulta os emails recentes não lidos ou importantes na Caixa de Entrada do Gmail.
        """
        if not self.is_google_configured():
            return []

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            token_path = Path("token.json")
            if not token_path.exists():
                return []

            creds = Credentials.from_authorized_user_file(str(token_path))
            service = build("gmail", "v1", credentials=creds)

            # Busca mensagens não lidas na caixa de entrada, filtrando promoções e redes sociais
            # Exclui categorias automatizadas comuns do Gmail: promotions, social, spam, trash
            query = "is:unread in:inbox -category:promotions -category:social -category:spam"
            results = await asyncio.to_thread(
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results * 4)
                .execute
            )
            messages = results.get("messages", [])
            parsed_emails = []

            for msg_ref in messages:
                if len(parsed_emails) >= max_results:
                    break

                msg_data = await asyncio.to_thread(
                    service.users().messages().get(userId="me", id=msg_ref["id"], format="full").execute
                )
                headers_raw = msg_data.get("payload", {}).get("headers", [])
                headers_dict = {h["name"].lower(): h["value"] for h in headers_raw}

                subject = headers_dict.get("subject", "Sem assunto")
                sender = headers_dict.get("from", "Desconhecido")
                date = headers_dict.get("date", "")
                snippet = msg_data.get("snippet", "")

                # Se o assunto contiver [IMPORTANTE] ou [URGENTE], tem prioridade e passa
                is_explicit_important = bool(re.search(r"\[?(importante|urgente)\]?", subject, re.IGNORECASE))

                if not is_explicit_important:
                    # 1. Filtra qualquer mailing list / newsletter (List-Unsubscribe, List-Id, etc.)
                    if any(h in headers_dict for h in ("list-unsubscribe", "list-id", "list-post", "list-owner", "precedence")):
                        continue

                    # 2. Filtra remetentes automáticos/newsletters/no-reply
                    sender_lower = sender.lower()
                    if any(re.search(pat, sender_lower) for pat in AUTOMATED_SENDER_PATTERNS):
                        continue

                    # 3. Filtra padrões de assuntos automatizados comuns
                    subj_lower = subject.lower()
                    if any(
                        k in subj_lower
                        for k in (
                            "security alert",
                            "alerta de segurança",
                            "security notification",
                            "got 5 minutes",
                            "the monthly build",
                            "black friday",
                            "desconto",
                            "cupom",
                            "termo atualizado",
                            "novas vagas semelhantes",
                            "está contratando:",
                            "recrutando agora",
                            "edição #",
                            "edicao #",
                            "boletim",
                            "newsletter",
                        )
                    ):
                        continue

                parsed_emails.append({
                    "id": msg_ref["id"],
                    "subject": subject,
                    "sender": sender,
                    "date": date,
                    "snippet": snippet,
                })

            return parsed_emails
        except Exception as e:
            logger.warning(f"[ProactiveHub] Gmail indisponível: {e}")
            return []

    # =========================================================================
    # 3. Disparador de Alertas para Barramento e Discord
    # =========================================================================

    async def broadcast_briefing_to_bus(self):
        """Publica o briefing compilado (Obsidian + Google Calendar + Gmail) no Redis para consumo de Voz e Interfaces."""
        briefing = self.get_upcoming_briefing()
        
        # Agrega Google Calendar se disponível
        g_events = await self.fetch_google_calendar_events(5)
        if g_events:
            # Formata eventos do Google Calendar com dia da semana e hora amigável
            DIAS_SEMANA = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
            g_lines = []
            for ev in g_events:
                dt_str = ev.get("date", "")
                label = ev["title"]
                try:
                    dt = datetime.fromisoformat(dt_str)
                    dia_nome = DIAS_SEMANA[dt.weekday()]
                    hora_min = dt.strftime("%H:%M")
                    dia_mes = dt.strftime("%d/%m")
                    g_lines.append(f"- 📅 {label} ({dia_nome}, {dia_mes} às {hora_min})")
                except Exception:
                    g_lines.append(f"- 📅 {label} ({dt_str})")

            briefing["summary_text"] += "\n\nGoogle Calendar:\n" + "\n".join(g_lines)
            briefing["items"].extend(g_events)
            briefing["total_pending"] += len(g_events)

        # Agrega Gmail se disponível
        g_emails = await self.fetch_unread_gmails(3)
        if g_emails:
            em_lines = [f"- ✉️ De: {em['sender']} | Assunto: {em['subject']}" for em in g_emails]
            briefing["summary_text"] += "\n\nEmails Recentes Não Lidos:\n" + "\n".join(em_lines)

        payload = {
            "type": "daily_briefing",
            "summary": briefing["summary_text"],
            "total_pending": briefing["total_pending"],
            "emails": g_emails,
            "timestamp": datetime.now().isoformat(),
        }
        await self.bus.publish(CH_PROACTIVE_ALERTS, payload)
        # Salva o último briefing no estado efêmero do Redis (válido por 12 horas)
        await self.bus.set_state("daily_briefing", payload, expire_seconds=43200)
        return briefing

    async def check_new_emails_and_notify(self):
        """
        Verifica se chegaram novos emails importantes ou diretos
        e publica no canal #aigis-inbox do Discord com alerta.
        """
        try:
            emails = await self.fetch_unread_gmails(max_results=5)
            notified_emails = self._load_notified_emails()

            # Resgata também histórico do Redis para sincronização redundante
            redis_notified = await self.bus.get_state("notified_emails") or []
            if isinstance(redis_notified, list):
                notified_emails.update(redis_notified)

            for em in emails:
                em_id = em["id"]
                if em_id in notified_emails:
                    continue

                subject = em.get("subject", "Sem assunto")
                sender = em.get("sender", "Desconhecido")
                snippet = em.get("snippet", "")

                # Detecta se é alta prioridade ou email direto
                is_urgent = any(k in subject.lower() for k in ("importante", "urgente", "vaga", "estágio", "estagio", "entrevista"))
                priority = "Alta" if is_urgent else "Normal"

                payload = {
                    "type": "inbox_alert",
                    "sender": sender,
                    "subject": subject,
                    "summary": snippet or "Novo email recebido na caixa de entrada.",
                    "date_str": em.get("date", ""),
                    "priority": priority,
                    "ping_owner": is_urgent,
                }
                await self.bus.publish(CH_PROACTIVE_ALERTS, payload)
                self._save_notified_email(em_id)
                notified_emails.add(em_id)
                await self.bus.set_state("notified_emails", list(notified_emails)[-100:])
                logger.info(f"[ProactiveHub] Alerta de novo email emitido no Discord: '{subject}'")

        except Exception as e:
            logger.debug(f"[ProactiveHub] Erro na checagem de emails: {e}")

    async def check_upcoming_reminders_and_ping(self):
        """
        Verifica se há algum compromisso começando nos próximos 15 a 30 minutos
        e envia um alerta de ping ativo no Discord.
        """
        try:
            from datetime import timedelta
            now = datetime.now()
            events = await self.fetch_google_calendar_events(max_results=5)
            notified_cache = await self.bus.get_state("notified_reminders") or []

            for ev in events:
                dt_str = ev.get("date", "")
                ev_title = ev.get("title", "Compromisso")
                ev_id = f"{ev_title}_{dt_str}"

                if ev_id in notified_cache:
                    continue

                try:
                    dt = datetime.fromisoformat(dt_str)
                    diff_min = (dt - now).total_seconds() / 60.0

                    # Se o evento começa entre 0 e 20 minutos a partir de agora:
                    if 0 <= diff_min <= 20:
                        payload = {
                            "type": "agenda_alert",
                            "title": f"🔔 Lembrete: {ev_title} em breve!",
                            "time_str": f"Em ~{int(diff_min)} minutos ({dt.strftime('%H:%M')})",
                            "details": ev.get("details") or "Seu próximo compromisso está prestes a começar.",
                            "location": ev.get("location", ""),
                            "ping_owner": True,
                        }
                        await self.bus.publish(CH_PROACTIVE_ALERTS, payload)
                        notified_cache.append(ev_id)
                        # Mantém cache dos últimos 50 lembretes disparados
                        await self.bus.set_state("notified_reminders", notified_cache[-50:], expire_seconds=86400)
                        logger.info(f"[ProactiveHub] Ping de lembrete emitido para '{ev_title}' ({int(diff_min)}m)")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[ProactiveHub] Erro na checagem de lembretes: {e}")

    async def sync_agenda_to_discord(self, discord_hub: Any):
        """Varre itens pendentes e publica novos eventos em #aigis-agenda."""
        briefing = self.get_upcoming_briefing()
        for item in briefing["items"][:3]:
            try:
                await discord_hub.post_agenda_event(
                    title=item["title"],
                    time_str=item.get("date") or "A definir",
                    details=item.get("raw_details", ""),
                    location=item.get("location", ""),
                    ping_owner=False,
                )
            except Exception as e:
                logger.error(f"[ProactiveHub] Erro ao postar no Discord: {e}")
