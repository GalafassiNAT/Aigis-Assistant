"""
Scanner e Monitor de Workspace Geral para a Aigis.
Agrega informações sobre:
1. Obsidian Vault (Notas Diárias, TCC, Metas, Ideias, Status)
2. Projetos em desenvolvimento em /home/nora/Projects (Branches Git, Últimos Commits, Dirty Files)
3. Google Calendar & Gmail (quando autenticado)
4. Perfil de hábitos e interesses do usuário compilado para a Aigis
"""

import asyncio
import os
import re
import subprocess
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
PROJECTS_DIR = Path(os.getenv("PROJECTS_PATH", "/home/nora/Projects"))


class WorkspaceHub:
    def __init__(self, vault_path: Path = VAULT_DIR, projects_path: Path = PROJECTS_DIR):
        self.vault_path = vault_path
        self.projects_path = projects_path
        self.bus = EventBus.get_instance()

    # =========================================================================
    # 1. Scanner de Projetos e Git Ativo
    # =========================================================================

    def scan_git_projects(self) -> list[dict[str, Any]]:
        """Varre os projetos locais no workspace e identifica repositórios ativos."""
        repos_info = []
        if not self.projects_path.exists():
            return repos_info

        for d in sorted(self.projects_path.iterdir()):
            if d.is_dir() and (d / ".git").exists():
                try:
                    # Branch atual
                    branch = subprocess.check_output(
                        ["git", "-C", str(d), "rev-parse", "--abbrev-ref", "HEAD"],
                        text=True,
                        timeout=2,
                    ).strip()

                    # Último commit
                    last_commit = subprocess.check_output(
                        ["git", "-C", str(d), "log", "-1", "--format=%h | %cd | %s", "--date=short"],
                        text=True,
                        timeout=2,
                    ).strip()

                    # Status (arquivos modificados não commitados)
                    dirty = subprocess.check_output(
                        ["git", "-C", str(d), "status", "--porcelain"],
                        text=True,
                        timeout=2,
                    ).strip()

                    repos_info.append({
                        "name": d.name,
                        "path": str(d),
                        "branch": branch,
                        "last_commit": last_commit,
                        "has_uncommitted": bool(dirty),
                    })
                except Exception as e:
                    logger.debug(f"[WorkspaceHub] Falha ao ler Git de {d.name}: {e}")

        return repos_info

    # =========================================================================
    # 2. Scanner de Notas Diárias & TCC no Obsidian
    # =========================================================================

    def scan_latest_daily_notes(self) -> dict[str, Any] | None:
        """Lê a nota diária mais recente para capturar foco, humor e metas."""
        daily_dir = self.vault_path / "Daily Notes"
        if not daily_dir.exists():
            return None

        notes = sorted(daily_dir.glob("*.md"))
        if not notes:
            return None

        latest = notes[-1]
        try:
            content = latest.read_text(encoding="utf-8")
            
            # Extrai Frontmatter básico
            humor_m = re.search(r"humor:\s*(.*)", content)
            energia_m = re.search(r"energia:\s*(.*)", content)
            
            # Extrai Foco
            foco_m = re.search(r"## 🎯 Foco de Hoje[^\n]*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
            foco = foco_m.group(1).strip() if foco_m else ""
            
            # Extrai TCC / Projetos
            tcc_m = re.search(r"## 🦅 Log TCC[^\n]*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
            tcc = tcc_m.group(1).strip() if tcc_m else ""

            return {
                "file": latest.name,
                "date": latest.stem.split()[0],
                "humor": humor_m.group(1).strip() if humor_m else "",
                "energia": energia_m.group(1).strip() if energia_m else "",
                "foco": foco,
                "log_tcc": tcc,
            }
        except Exception as e:
            logger.warning(f"[WorkspaceHub] Erro ao ler Daily Note {latest.name}: {e}")
            return None

    # =========================================================================
    # 3. Scanner de Interesses & Ideias de Pesquisa (Aigis, Redes Neurais, Mandarim)
    # =========================================================================

    def scan_user_topics_and_studies(self) -> list[str]:
        """Varre tópicos de estudo ativos (Chinese, Studying, NOS, Livros)."""
        topics = []
        if not self.vault_path.exists():
            return topics

        study_dir = self.vault_path / "Studying"
        if study_dir.exists():
            for folder in study_dir.iterdir():
                if folder.is_dir():
                    topics.append(f"Estudo: {folder.name}")

        proj_aigis = self.vault_path / "Projetos" / "Project AIGIS"
        if proj_aigis.exists():
            topics.append("Projeto Ativo: Project AIGIS (Experiência Sensorial e Abstração Emocional)")

        return topics

    # =========================================================================
    # 4. Geração de Snapshot do Workspace para a Aigis
    # =========================================================================

    def build_workspace_snapshot(self) -> dict[str, Any]:
        """Compila a visão holística do ambiente de desenvolvimento e rotina do usuário."""
        repos = self.scan_git_projects()
        daily = self.scan_latest_daily_notes()
        topics = self.scan_user_topics_and_studies()

        # Resumo legível para o Prompt de Sistema da Aigis
        summary_lines = []
        if daily:
            summary_lines.append(f"Última Daily Note ({daily['date']}): Foco: {daily['foco'] or 'Geral'}")
            if daily['log_tcc']:
                summary_lines.append(f"Andamento TCC: {daily['log_tcc'][:120]}")

        active_projects = [f"{r['name']} ({r['branch']})" for r in repos]
        if active_projects:
            summary_lines.append(f"Projetos Ativos no PC: {', '.join(active_projects)}")

        if topics:
            summary_lines.append(f"Áreas de Interesse/Estudo: {', '.join(topics[:4])}")

        return {
            "summary_text": "\n".join(summary_lines),
            "projects": repos,
            "daily_note": daily,
            "topics": topics,
            "updated_at": datetime.now().isoformat(),
        }

    async def update_workspace_state(self):
        """Atualiza o estado em memória no Redis para que LLM e Voz acessem instantaneamente."""
        snapshot = self.build_workspace_snapshot()
        await self.bus.set_state("user_workspace_context", snapshot, expire_seconds=86400)
        return snapshot
