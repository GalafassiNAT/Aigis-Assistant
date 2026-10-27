import asyncio
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://ai_user:ai_password@localhost:5433/ai_assistant"
)


# --- MODELOS DE DADOS (DOMÍNIO) ---


class MemoryRecord(BaseModel):
    """Representa um fragmento de memória semântica recuperado do banco."""

    id: int
    user_id: str = "local:owner"
    content: str
    similarity: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class KnowledgeTriplet(BaseModel):
    """Representa uma relação atômica no Grafo de Conhecimento."""

    id: int
    user_id: str = "local:owner"
    subject: str
    predicate: str
    object: str
    created_at: datetime | None = None


# --- REPOSITÓRIO DE MEMÓRIA (MULTI-USER) ---


class MemoryStorage:
    def __init__(self, db_url: str = DEFAULT_DB_URL):
        self.db_url = db_url

    def _get_connection(self):
        conn = psycopg.connect(self.db_url, connect_timeout=2)
        register_vector(conn)
        return conn

    # --- MEMÓRIAS VETORIAIS (SEMÂNTICAS) ---

    def _sync_add_memory(
        self,
        content: str,
        embedding: list[float] | np.ndarray | None,
        metadata: dict | None,
        user_id: str,
    ) -> int:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        vec = np.array(embedding, dtype=np.float32) if embedding is not None else None

        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (content, metadata, embedding, user_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (content, metadata_json, vec, user_id),
            )
            mem_id = cur.fetchone()[0]
            conn.commit()
            return mem_id

    async def add_memory(
        self,
        content: str,
        embedding: list[float] | np.ndarray | None = None,
        metadata: dict | None = None,
        user_id: str = "local:owner",
    ) -> int:
        """Salva um texto e seu embedding vetorial atrelados a um usuário."""
        return await asyncio.to_thread(self._sync_add_memory, content, embedding, metadata, user_id)

    def _sync_search_memories(
        self,
        query_embedding: list[float] | np.ndarray,
        limit: int = 5,
        min_similarity: float = 0.25,
        user_id: str = "local:owner",
    ) -> list[MemoryRecord]:
        vec = np.array(query_embedding, dtype=np.float32)
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata, 1 - (embedding <=> %s) AS similarity, created_at, user_id
                FROM memories
                WHERE (user_id = %s OR user_id = 'local:owner')
                  AND embedding IS NOT NULL
                  AND (1 - (embedding <=> %s)) >= %s
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (vec, user_id, vec, min_similarity, vec, limit),
            )
            rows = cur.fetchall()
            results = []
            for row in rows:
                meta = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
                results.append(
                    MemoryRecord(
                        id=row[0],
                        content=row[1],
                        metadata=meta,
                        similarity=round(float(row[3]), 4),
                        created_at=row[4],
                        user_id=row[5] or user_id,
                    )
                )
            return results

    async def search_memories(
        self,
        query_embedding: list[float] | np.ndarray,
        limit: int = 5,
        min_similarity: float = 0.25,
        user_id: str = "local:owner",
    ) -> list[MemoryRecord]:
        """Busca as memórias do usuário mais próximas semanticamente usando distância de cosseno (<=>)."""
        return await asyncio.to_thread(
            self._sync_search_memories, query_embedding, limit, min_similarity, user_id
        )

    # --- GRAFO DE CONHECIMENTO (TRIPLETOS) ---

    def _sync_add_triplet(self, subject: str, predicate: str, object_: str, user_id: str) -> int:
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_triplets (subject, predicate, object, user_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (subject.strip(), predicate.strip(), object_.strip(), user_id),
            )
            triplet_id = cur.fetchone()[0]
            conn.commit()
            return triplet_id

    async def add_triplet(
        self, subject: str, predicate: str, object_: str, user_id: str = "local:owner"
    ) -> int:
        """Adiciona um fato estruturado ao grafo atrelado ao usuário."""
        return await asyncio.to_thread(self._sync_add_triplet, subject, predicate, object_, user_id)

    def _sync_add_triplets(self, triplets: list[dict], user_id: str) -> list[int]:
        """Grava múltiplos fatos em uma única transação atômica."""
        ids = []
        with self._get_connection() as conn, conn.cursor() as cur:
            for t in triplets:
                cur.execute(
                    """
                    INSERT INTO knowledge_triplets (subject, predicate, object, user_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (t["subject"].strip(), t["predicate"].strip(), t["object"].strip(), user_id),
                )
                ids.append(cur.fetchone()[0])
            conn.commit()
        return ids

    async def add_triplets(self, triplets: list[dict], user_id: str = "local:owner") -> list[int]:
        """Adiciona uma lista de fatos estruturados de uma só vez."""
        return await asyncio.to_thread(self._sync_add_triplets, triplets, user_id)

    def _sync_search_triplets(self, keyword: str, user_id: str) -> list[KnowledgeTriplet]:
        pattern = f"%{keyword.strip()}%"
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, subject, predicate, object, created_at, user_id
                FROM knowledge_triplets
                WHERE (user_id = %s OR user_id = 'local:owner')
                  AND (subject ILIKE %s OR predicate ILIKE %s OR object ILIKE %s)
                ORDER BY created_at DESC;
                """,
                (user_id, pattern, pattern, pattern),
            )
            rows = cur.fetchall()
            return [
                KnowledgeTriplet(
                    id=r[0],
                    subject=r[1],
                    predicate=r[2],
                    object=r[3],
                    created_at=r[4],
                    user_id=r[5] or user_id,
                )
                for r in rows
            ]

    async def search_triplets(
        self, keyword: str, user_id: str = "local:owner"
    ) -> list[KnowledgeTriplet]:
        """Busca relações no grafo que contenham a palavra-chave informada para o usuário."""
        return await asyncio.to_thread(self._sync_search_triplets, keyword, user_id)

    def _sync_get_all_triplets(self, user_id: str) -> list[KnowledgeTriplet]:
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, subject, predicate, object, created_at, user_id
                FROM knowledge_triplets
                WHERE (user_id = %s OR user_id = 'local:owner')
                ORDER BY id ASC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            return [
                KnowledgeTriplet(
                    id=r[0],
                    subject=r[1],
                    predicate=r[2],
                    object=r[3],
                    created_at=r[4],
                    user_id=r[5] or user_id,
                )
                for r in rows
            ]

    async def get_all_triplets(self, user_id: str = "local:owner") -> list[KnowledgeTriplet]:
        """Retorna todos os fatos do grafo associados ao usuário."""
        return await asyncio.to_thread(self._sync_get_all_triplets, user_id)
