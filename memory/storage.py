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


class UserProfileTrait(BaseModel):
    """Representa um traço de perfil, hábito ou estilo do usuário ou contato."""

    id: int
    user_id: str = "local:owner"
    category: str
    attribute: str
    description: str
    confidence: float = 1.0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class EpisodicMemory(BaseModel):
    """Representa uma memória autobiográfica datada de eventos ou sessões compartilhadas."""

    id: int
    user_id: str = "local:owner"
    title: str
    summary: str
    emotional_tone: str | None = "neutro"
    key_topics: list[str] = Field(default_factory=list)
    occurred_on: str | None = None
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

    # --- PERFIL DO USUÁRIO & DINÂMICA SOCIAL ---

    def _sync_upsert_user_profile(
        self,
        category: str,
        attribute: str,
        description: str,
        confidence: float,
        user_id: str,
    ) -> int:
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_profiles (user_id, category, attribute, description, confidence, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, category, attribute)
                DO UPDATE SET
                    description = EXCLUDED.description,
                    confidence = EXCLUDED.confidence,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
                """,
                (user_id, category.strip().lower(), attribute.strip().lower(), description.strip(), confidence),
            )
            trait_id = cur.fetchone()[0]
            conn.commit()
            return trait_id

    async def upsert_user_profile(
        self,
        category: str,
        attribute: str,
        description: str,
        confidence: float = 1.0,
        user_id: str = "local:owner",
    ) -> int:
        """Grava ou atualiza um traço comportamental, hábito ou estilo do usuário/contato."""
        return await asyncio.to_thread(
            self._sync_upsert_user_profile, category, attribute, description, confidence, user_id
        )

    def _sync_get_user_profile(self, user_id: str, category: str | None = None) -> list[UserProfileTrait]:
        with self._get_connection() as conn, conn.cursor() as cur:
            if category:
                cur.execute(
                    """
                    SELECT id, user_id, category, attribute, description, confidence, created_at, updated_at
                    FROM user_profiles
                    WHERE (user_id = %s OR user_id = 'local:owner') AND category = %s
                    ORDER BY confidence DESC, updated_at DESC;
                    """,
                    (user_id, category.strip().lower()),
                )
            else:
                cur.execute(
                    """
                    SELECT id, user_id, category, attribute, description, confidence, created_at, updated_at
                    FROM user_profiles
                    WHERE (user_id = %s OR user_id = 'local:owner')
                    ORDER BY category ASC, confidence DESC, updated_at DESC;
                    """,
                    (user_id,),
                )
            rows = cur.fetchall()
            return [
                UserProfileTrait(
                    id=r[0],
                    user_id=r[1],
                    category=r[2],
                    attribute=r[3],
                    description=r[4],
                    confidence=float(r[5]),
                    created_at=r[6],
                    updated_at=r[7],
                )
                for r in rows
            ]

    async def get_user_profile(
        self, user_id: str = "local:owner", category: str | None = None
    ) -> list[UserProfileTrait]:
        """Recupera o perfil estruturado (hábitos, preferências, estilo) do usuário."""
        return await asyncio.to_thread(self._sync_get_user_profile, user_id, category)

    # --- MEMÓRIAS EPISÓDICAS & LINHA DO TEMPO AUTOBIOGRÁFICA ---

    def _sync_add_episodic_memory(
        self,
        title: str,
        summary: str,
        emotional_tone: str,
        key_topics: list[str],
        user_id: str,
    ) -> int:
        topics_json = json.dumps(key_topics or [], ensure_ascii=False)
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodic_memories (user_id, title, summary, emotional_tone, key_topics, occurred_on)
                VALUES (%s, %s, %s, %s, %s, CURRENT_DATE)
                RETURNING id;
                """,
                (user_id, title.strip(), summary.strip(), emotional_tone.strip(), topics_json),
            )
            ep_id = cur.fetchone()[0]
            conn.commit()
            return ep_id

    async def add_episodic_memory(
        self,
        title: str,
        summary: str,
        emotional_tone: str = "neutro",
        key_topics: list[str] | None = None,
        user_id: str = "local:owner",
    ) -> int:
        """Salva uma memória episódica autobiográfica (resumo de uma sessão ou dia)."""
        return await asyncio.to_thread(
            self._sync_add_episodic_memory, title, summary, emotional_tone, key_topics or [], user_id
        )

    def _sync_get_recent_episodes(self, limit: int, user_id: str) -> list[EpisodicMemory]:
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, summary, emotional_tone, key_topics, occurred_on, created_at
                FROM episodic_memories
                WHERE (user_id = %s OR user_id = 'local:owner')
                ORDER BY occurred_on DESC, id DESC
                LIMIT %s;
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
            return [
                EpisodicMemory(
                    id=r[0],
                    user_id=r[1],
                    title=r[2],
                    summary=r[3],
                    emotional_tone=r[4],
                    key_topics=r[5] if isinstance(r[5], list) else json.loads(r[5] or "[]"),
                    occurred_on=str(r[6]),
                    created_at=r[7],
                )
                for r in rows
            ]

    async def get_recent_episodes(
        self, limit: int = 5, user_id: str = "local:owner"
    ) -> list[EpisodicMemory]:
        """Recupera os episódios recentes da história compartilhada com o usuário."""
        return await asyncio.to_thread(self._sync_get_recent_episodes, limit, user_id)
