import asyncio
import json
import os
import sys
from pathlib import Path

# Garante path da raiz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from openai import AsyncOpenAI

from memory.embeddings import get_embedding
from memory.storage import MemoryStorage

load_dotenv()

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080/v1")
MODEL_ALIAS = os.getenv("LLAMA_MODEL_ALIAS", "qwen-coder")
OWNER_DISCORD_ID = os.getenv("DISCORD_OWNER_ID", "").strip()

client = AsyncOpenAI(base_url=LLAMA_SERVER_URL, api_key="no_need")
storage = MemoryStorage()

EXTRACTION_SYSTEM_PROMPT = """Você é um extrator de conhecimento e memória para uma IA assistente técnica.
Sua função é analisar uma conversa recente entre o Usuário e a IA e extrair fatos duradouros, preferências ou correções.

Regra Fundamental:
- Você deve extrair EXCLUSIVAMENTE o que o USUÁRIO declarou, ensinou ou corrigiu como verdade.
- NUNCA extraia alucinações, palpites incorretos ou erros que o Assistente disse e que o usuário contestou ou corrigiu em seguida.
- Se o usuário disser "Não, isso está errado", "não é bem isso", ou "sai um por semana", DESCARTE completamente a afirmação anterior do Assistente e considere apenas o que o Usuário instruiu.

Diretrizes:
1. Extraia APENAS fatos novos ensinados pelo usuário, decisões de projeto ou correções explícitas feitas pelo usuário.
2. NUNCA extraia saudações ("olá", "como vai"), mensagens triviais ou especulações.
3. Para cada fato válido ensinado pelo usuário, gere um objeto com:
   - "subject": Sujeito (ex: 'RLOS', 'Aigis', 'Natanael')
   - "predicate": Verbo/Relação em minúsculas (ex: 'é', 'desenvolve', 'usa', 'formato')
   - "object": O valor, especificação ou descrição concisa informada pelo usuário.
   - "is_correction": true se for uma correção explícita feita pelo usuário sobre um erro anterior, false caso contrário.

Responda ESTRITAMENTE em formato JSON:
{"facts": [{"subject": "...", "predicate": "...", "object": "...", "is_correction": false}]}
Se não houver nenhum fato duradouro na conversa, retorne: {"facts": []}
"""


def _get_worker_state() -> int:
    with storage._get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT last_processed_log_id FROM memory_worker_state WHERE id = 'chat_logs_extractor';"
        )
        row = cur.fetchone()
        return row[0] if row else 0


def _update_worker_state(last_id: int):
    with storage._get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE memory_worker_state SET last_processed_log_id = %s, updated_at = CURRENT_TIMESTAMP WHERE id = 'chat_logs_extractor';",
            (last_id,),
        )
        conn.commit()


def _get_unprocessed_logs(last_id: int, limit: int = 15) -> list[dict]:
    with storage._get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, channel_name, author_id, author_name, content, is_bot, created_at
            FROM chat_logs
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
                "channel_name": r[1],
                "author_id": r[2],
                "author_name": r[3],
                "content": r[4],
                "is_bot": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]


async def extract_facts_from_messages(dialogue_text: str) -> list[dict]:
    """Usa o Qwen local para extrair tripletos atômicos estruturados da conversa."""
    try:
        resp = await client.chat.completions.create(
            model=MODEL_ALIAS,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analise a seguinte conversa recente:\n\n{dialogue_text}",
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )
        content = resp.choices[0].message.content or ""
        # Extrai bloco JSON
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            dados = json.loads(content[brace_start : brace_end + 1])
            return dados.get("facts", [])
    except Exception as e:
        print(f"Erro na extração de fatos via LLM: {e}")
    return []


def consolidate_fact(fact: dict, target_user_id: str):
    """
    Grava fato no Grafo e na busca vetorial.
    Se for uma correção, remove ou sobrescreve versões conflitantes anteriores.
    """
    subj = fact.get("subject", "").strip()
    pred = fact.get("predicate", "").strip()
    obj = fact.get("object", "").strip()
    is_corr = bool(fact.get("is_correction", False))

    if not subj or not pred or not obj:
        return

    with storage._get_connection() as conn, conn.cursor() as cur:
        # 1. Se for correção, apaga tripletos anteriores do mesmo subject e predicate
        if is_corr:
            cur.execute(
                """
                DELETE FROM knowledge_triplets
                WHERE user_id = %s AND subject ILIKE %s AND predicate ILIKE %s;
                """,
                (target_user_id, subj, pred),
            )

        # 2. Insere no Grafo
        cur.execute(
            """
            INSERT INTO knowledge_triplets (subject, predicate, object, user_id)
            VALUES (%s, %s, %s, %s);
            """,
            (subj, pred, obj, target_user_id),
        )
        conn.commit()

    # 3. Gera embedding e insere em memories
    frase = f"{subj} {pred} {obj}"
    vec = get_embedding(frase)
    storage._sync_add_memory(
        content=frase,
        embedding=vec,
        metadata={"source": "autonomous_consolidation", "is_correction": is_corr},
        user_id=target_user_id,
    )
    print(
        f"🧠 [Consolidação] Gravado: [{subj}] -> [{pred}] -> [{obj}] (user={target_user_id}, correcao={is_corr})"
    )


async def run_consolidation_cycle():
    """Executa um ciclo único de consolidação de memórias acumuladas."""
    last_id = _get_worker_state()
    logs = _get_unprocessed_logs(last_id, limit=20)

    if not logs:
        return 0

    print(f"🔍 [Worker] Processando {len(logs)} mensagens a partir do log #{last_id + 1}...")

    # Monta texto do diálogo
    dialogo_linhas = []
    has_owner_msg = False
    for log_item in logs:
        papel = "Assistente" if log_item["is_bot"] else f"Usuário ({log_item['author_name']})"
        dialogo_linhas.append(f"{papel}: {log_item['content']}")
        if log_item["author_id"] == OWNER_DISCORD_ID:
            has_owner_msg = True

    dialogo_str = "\n".join(dialogo_linhas)
    target_user_id = "local:owner" if has_owner_msg else f"discord:{logs[0]['author_id']}"

    fatos = await extract_facts_from_messages(dialogo_str)
    for f in fatos:
        consolidate_fact(f, target_user_id)

    max_id = max(log_item["id"] for log_item in logs)
    _update_worker_state(max_id)
    print(f"✅ [Worker] Ciclo concluído até log #{max_id}. {len(fatos)} fatos consolidados.")
    return len(fatos)


async def main_loop(poll_interval: int = 30):
    """Loop contínuo de background rodando a cada poll_interval segundos."""
    print(f"🚀 Background Memory Worker iniciado (intervalo: {poll_interval}s)")
    while True:
        try:
            await run_consolidation_cycle()
        except Exception as e:
            print(f"⚠️ Erro no loop de consolidação: {e}")
        await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(main_loop(poll_interval=20))
