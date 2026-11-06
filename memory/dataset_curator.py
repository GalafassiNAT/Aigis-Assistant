import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

TRAINING_DIR = Path("data/training")
SFT_DATASET_FILE = TRAINING_DIR / "aigis_sft.jsonl"
DPO_DATASET_FILE = TRAINING_DIR / "aigis_dpo.jsonl"


class DatasetCurator:
    """
    Curador autônomo de dados para evolução contínua dos pesos da Aigis.
    Coleta diálogos reais de alta qualidade (SFT) e correções de preferência (DPO)
    para alimentar pipelines futuros de Fine-Tuning (QLoRA / Unsloth).
    """

    _instance: "DatasetCurator | None" = None

    def __init__(self, training_dir: Path = TRAINING_DIR):
        self.training_dir = training_dir
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.sft_file = SFT_DATASET_FILE
        self.dpo_file = DPO_DATASET_FILE

    @classmethod
    def get_instance(cls) -> "DatasetCurator":
        if cls._instance is None:
            cls._instance = DatasetCurator()
        return cls._instance

    def record_sft_dialogue(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Salva um diálogo bem-sucedido no formato ChatML padrão para Supervised Fine-Tuning (SFT).
        Ignora saudações vazias ou turnos sem substância.
        """
        if not messages or len(messages) < 2:
            return False

        # Verifica se há pelo menos um turno substancial de usuário e assistente
        user_texts = [m["content"] for m in messages if m.get("role") == "user"]
        assistant_texts = [m["content"] for m in messages if m.get("role") == "assistant"]

        if not user_texts or not assistant_texts:
            return False

        # Evita conversas triviais de 1 ou 2 palavras
        total_len = sum(len(t) for t in user_texts + assistant_texts)
        if total_len < 30:
            return False

        record = {
            "messages": messages,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(self.sft_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug(f"[DatasetCurator] SFT diálogo salvo ({len(messages)} mensagens).")
            return True
        except Exception as e:
            logger.error(f"[DatasetCurator] Erro ao salvar SFT record: {e}")
            return False

    def record_dpo_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        reason: str = "",
    ) -> bool:
        """
        Salva um par de alinhamento DPO (Direct Preference Optimization).
        - chosen: A resposta ideal / corrigida que a Aigis deveria dar.
        - rejected: A resposta falha, alucinada ou mecânica anterior.
        """
        if not prompt or not chosen or not rejected or chosen == rejected:
            return False

        record = {
            "prompt": prompt.strip(),
            "chosen": chosen.strip(),
            "rejected": rejected.strip(),
            "reason": reason.strip(),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(self.dpo_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"[DatasetCurator] Par DPO registrado! Razão: '{reason}'")
            return True
        except Exception as e:
            logger.error(f"[DatasetCurator] Erro ao salvar DPO record: {e}")
            return False

    def get_dataset_summary(self) -> dict[str, int]:
        """Retorna a contagem de amostras coletadas para treinamento."""
        sft_count = 0
        dpo_count = 0

        if self.sft_file.exists():
            try:
                with open(self.sft_file, "r", encoding="utf-8") as f:
                    sft_count = sum(1 for line in f if line.strip())
            except Exception:
                pass

        if self.dpo_file.exists():
            try:
                with open(self.dpo_file, "r", encoding="utf-8") as f:
                    dpo_count = sum(1 for line in f if line.strip())
            except Exception:
                pass

        return {
            "sft_samples": sft_count,
            "dpo_samples": dpo_count,
            "total": sft_count + dpo_count,
        }
