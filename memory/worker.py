import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Garante path da raiz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from core.persona import PersonaManager
from memory.dataset_curator import DatasetCurator
from memory.embeddings import get_embedding
from memory.storage import MemoryStorage

load_dotenv()

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080/v1")
MODEL_ALIAS = os.getenv("LLAMA_MODEL_ALIAS", "qwen-coder")

client = AsyncOpenAI(base_url=LLAMA_SERVER_URL, api_key="no_need")
storage = MemoryStorage()
persona_mgr = PersonaManager.get_instance()
curator = DatasetCurator.get_instance()

COGNITIVE_EXTRACTION_PROMPT = """Você é o motor cognitivo e de auto-reflexão da Aigis (IA assistente pessoal de Natanael).
Sua função é analisar os diálogos recentes entre Natanael (ou outros usuários) e a Aigis e extrair conhecimento multidimensional para a evolução da assistente.

Você deve extrair 5 dimensões em um único JSON estruturado:

1. "facts": Fatos objetivos duradouros (ex: projetos, ferramentas, fatos técnicos).
   - "subject", "predicate", "object", "is_correction" (true se corrigiu um erro anterior).
2. "profile_traits": Hábitos, preferências subjetivas, ritmo biológico, opiniões ou dinâmica social do usuário.
   - "category": 'habito', 'preferencia', 'estilo_pensamento', 'estado_emocional', 'relacao_social', 'meta_pessoal'.
   - "attribute": Identificador conciso (ex: 'rotina_estudos', 'tom_preferido', 'sono').
   - "description": Descrição natural do comportamento ou preferência observada.
   - "confidence": Float de 0.5 a 1.0.
3. "episode": Se o diálogo formou uma sessão coerente, resolução de problema ou marco importante, gere um resumo episódico. Caso contrário, null.
   - "title": Título curto (ex: 'Configuração da rotina e ajuste de emails').
   - "summary": Resumo conciso do que viveram juntos.
   - "emotional_tone": 'produtivo', 'tranquilo', 'desafiador', 'divertido'.
   - "key_topics": Lista de palavras-chave.
4. "dpo_correction": Se o usuário deu uma bronca ou corrigiu explicitamente uma falha/alucinação da Aigis, capture o par para alinhamento. Caso contrário, null.
   - "prompt": O que o usuário havia pedido antes do erro.
   - "chosen": O comportamento ou resposta correta esperada.
   - "rejected": O erro cometido pela Aigis que foi contestado.
   - "reason": Por que a resposta anterior estava errada.
5. "persona_learning": Uma diretriz prática aprendida para aprimorar o comportamento da Aigis no futuro. Se não houver, null.

Responda ESTRITAMENTE em formato JSON:
{
  "facts": [],
  "profile_traits": [],
  "episode": null,
  "dpo_correction": null,
  "persona_learning": null
}
"""


def _get_worker_state() -> int:
    with storage._get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT last_processed_id FROM memory_worker_state WHERE id = 'timeline_cognitive_worker';"
        )
        row = cur.fetchone()
        return row[0] if row else 0


def _update_worker_state(last_id: int):
    with storage._get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO memory_worker_state (id, last_processed_id, updated_at)
            VALUES ('timeline_cognitive_worker', %s, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO UPDATE SET
                last_processed_id = EXCLUDED.last_processed_id,
                updated_at = CURRENT_TIMESTAMP;
            """,
            (last_id,),
        )
        conn.commit()


def _get_unprocessed_timeline(last_id: int, limit: int = 15) -> list[dict[str, Any]]:
    with storage._get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, author, content, metadata, created_at
            FROM shared_timeline
            WHERE id > %s
            ORDER BY id ASC
            LIMIT %s;
            """,
            (last_id, limit),
        )
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "source": r[1],
                "author": r[2],
                "content": r[3],
                "metadata": r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}"),
                "created_at": r[5],
            }
            for r in rows
        ]


async def extract_cognitive_insights(dialogue_text: str) -> dict[str, Any]:
    """Chama o LLM local para extrair perfil, fatos, episódios e alinhamentos."""
    try:
        resp = await client.chat.completions.create(
            model=MODEL_ALIAS,
            messages=[
                {"role": "system", "content": COGNITIVE_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Analise o seguinte trecho da linha do tempo recente:\n\n{dialogue_text}",
                },
            ],
            temperature=0.1,
            max_tokens=800,
        )
        content = resp.choices[0].message.content or ""
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            return json.loads(content[brace_start : brace_end + 1])
    except Exception as e:
        logger.warning(f"[CognitiveWorker] Erro na extração via LLM: {e}")
    return {}


async def process_timeline_batch():
    """Executa um ciclo de processamento da linha do tempo compartilhada."""
    last_id = _get_worker_state()
    events = _get_unprocessed_timeline(last_id, limit=12)

    if not events:
        return 0

    logger.info(f"[CognitiveWorker] Processando {len(events)} eventos a partir do id #{last_id + 1}...")

    dialogue_lines = []
    messages_for_sft = []

    for ev in events:
        author = ev["author"]
        content = ev["content"]
        is_assistant = author.lower() in ("aigis", "assistant")
        role = "assistant" if is_assistant else "user"

        dialogue_lines.append(f"{author}: {content}")
        messages_for_sft.append({"role": role, "content": content})

    dialogue_str = "\n".join(dialogue_lines)
    insights = await extract_cognitive_insights(dialogue_str)

    target_user_id = "local:owner"

    # 1. Consolida fatos no Grafo e vetores
    facts = insights.get("facts", [])
    for f in facts:
        subj = f.get("subject", "").strip()
        pred = f.get("predicate", "").strip()
        obj = f.get("object", "").strip()
        is_corr = bool(f.get("is_correction", False))

        if subj and pred and obj:
            if is_corr:
                with storage._get_connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM knowledge_triplets WHERE user_id = %s AND subject ILIKE %s AND predicate ILIKE %s;",
                        (target_user_id, subj, pred),
                    )
                    conn.commit()

            await storage.add_triplet(subj, pred, obj, user_id=target_user_id)
            phrase = f"{subj} {pred} {obj}"
            vec = get_embedding(phrase)
            await storage.add_memory(
                content=phrase,
                embedding=vec,
                metadata={"source": "cognitive_worker", "is_correction": is_corr},
                user_id=target_user_id,
            )
            logger.info(f"🧠 [Fato Consolidado]: {subj} -> {pred} -> {obj}")

    # 2. Consolida perfil psicológico, hábitos e preferências
    traits = insights.get("profile_traits", [])
    for t in traits:
        cat = t.get("category", "habito")
        attr = t.get("attribute", "")
        desc = t.get("description", "")
        conf = float(t.get("confidence", 1.0))
        if attr and desc:
            await storage.upsert_user_profile(
                category=cat,
                attribute=attr,
                description=desc,
                confidence=conf,
                user_id=target_user_id,
            )
            logger.info(f"👤 [Perfil Absorvido] ({cat}): {attr} -> {desc}")

    # 3. Consolida memória episódica autobiográfica
    ep = insights.get("episode")
    if ep and isinstance(ep, dict) and ep.get("title") and ep.get("summary"):
        await storage.add_episodic_memory(
            title=ep["title"],
            summary=ep["summary"],
            emotional_tone=ep.get("emotional_tone", "neutro"),
            key_topics=ep.get("key_topics", []),
            user_id=target_user_id,
        )
        logger.info(f"📖 [Episódio Autobiográfico]: '{ep['title']}'")

    # 4. Registra correções DPO se houver
    dpo = insights.get("dpo_correction")
    if dpo and isinstance(dpo, dict) and dpo.get("prompt") and dpo.get("chosen") and dpo.get("rejected"):
        curator.record_dpo_pair(
            prompt=dpo["prompt"],
            chosen=dpo["chosen"],
            rejected=dpo["rejected"],
            reason=dpo.get("reason", "Correção explícita de usuário"),
        )

    # 5. Evolui o estado de persona se houve aprendizado comportamental
    persona_learning = insights.get("persona_learning")
    if persona_learning and isinstance(persona_learning, str) and len(persona_learning) > 10:
        persona_mgr.add_learned_trait(persona_learning)
        logger.info(f"✨ [Evolução de Persona]: {persona_learning}")

    # 6. Salva diálogo no dataset SFT se for construtivo e sem correções
    if not dpo and len(messages_for_sft) >= 2:
        curator.record_sft_dialogue(messages_for_sft, metadata={"source": "shared_timeline"})

    max_id = max(ev["id"] for ev in events)
    _update_worker_state(max_id)
    logger.info(f"✅ [CognitiveWorker] Lote finalizado até id #{max_id}.")
    return len(events)


async def main_loop(poll_interval: int = 30):
    """Loop contínuo de background rodando a cada poll_interval segundos."""
    logger.info(f"🚀 Cognitive Reflection Worker iniciado (intervalo: {poll_interval}s)")
    while True:
        try:
            await process_timeline_batch()
        except Exception as e:
            logger.error(f"[CognitiveWorker] Erro no ciclo de reflexão: {e}")
        await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(main_loop(poll_interval=20))
