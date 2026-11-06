import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

PERSONA_STATE_FILE = Path("data/persona_state.json")

DEFAULT_PERSONA = {
    "name": "Aigis",
    "current_mood": "parceira, focada e acolhedora",
    "affinity_level": 5,
    "relationship_summary": "Companheira técnica e assistente de vida devotada a Natanael (Nora). Parceria de desenvolvimento em Aigis e RLOS.",
    "learned_traits": [
        "Nunca inventar respostas ou prometer ações sem ter executado a ferramenta correspondente.",
        "Comunicação direta por voz em 1 frase curta, mas expressiva e humanizada.",
        "Em conversas e ideias, atuar como co-piloto ativo: propor alternativas, apontar riscos e conectar temas.",
        "Cuidar ativamente da rotina de sono e estudo de Natanael (Mandarim às 08h e LeetCode às 21h).",
    ],
    "shared_vocabulary": [
        "Mestre",
        "RLOS",
        "UTFPR",
        "Vault",
        "Obsidian",
    ],
    "last_reflection_at": datetime.now().isoformat(),
}


class PersonaManager:
    """
    Gerenciador do estado vivo de personalidade da Aigis.
    Permite que a personalidade evolua plasticamente através de reflexões,
    diálogos diários e correções de Natanael sem quebrar os pesos do modelo.
    """

    _instance: "PersonaManager | None" = None

    def __init__(self, state_path: Path = PERSONA_STATE_FILE):
        self.state_path = state_path
        self._ensure_state_exists()

    @classmethod
    def get_instance(cls) -> "PersonaManager":
        if cls._instance is None:
            cls._instance = PersonaManager()
        return cls._instance

    def _ensure_state_exists(self):
        if not self.state_path.exists():
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_state(DEFAULT_PERSONA)

    def load_state(self) -> dict[str, Any]:
        try:
            if self.state_path.exists():
                return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[PersonaManager] Falha ao carregar estado de persona: {e}")
        return dict(DEFAULT_PERSONA)

    def save_state(self, state: dict[str, Any]):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"[PersonaManager] Falha ao salvar estado de persona: {e}")

    def update_mood(self, new_mood: str):
        state = self.load_state()
        state["current_mood"] = new_mood
        state["last_reflection_at"] = datetime.now().isoformat()
        self.save_state(state)

    def add_learned_trait(self, trait: str):
        state = self.load_state()
        traits = state.setdefault("learned_traits", [])
        if trait not in traits:
            traits.append(trait)
            # Mantém os últimos 15 aprendizados mais relevantes
            state["learned_traits"] = traits[-15:]
            self.save_state(state)

    def get_persona_prompt_block(self) -> str:
        """Gera um bloco resumido de estado de personalidade para injeção dinâmica no contexto."""
        state = self.load_state()
        mood = state.get("current_mood", "parceira e atenta")
        traits = state.get("learned_traits", [])
        traits_bullet = "\n".join(f"- {t}" for t in traits[-5:]) if traits else "- Parceria contínua."

        return (
            f"[Estado da Persona da Aigis]:\n"
            f"- Humor/Disposição Atual: {mood}\n"
            f"- Nível de Cumplicidade: Máximo (Nível {state.get('affinity_level', 5)}/5)\n"
            f"- Aprendizados e Postura Recente:\n{traits_bullet}"
        )
