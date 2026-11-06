-- Ativa a extensão vetorial
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabela de memórias semânticas (vetores)
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) DEFAULT 'local:owner',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de fatos do Grafo de Conhecimento
CREATE TABLE IF NOT EXISTS knowledge_triplets (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) DEFAULT 'local:owner',
    subject VARCHAR(255) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    object VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de Perfis Psicológicos, Hábitos e Dinâmica Social
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL DEFAULT 'local:owner',
    category VARCHAR(50) NOT NULL,
    attribute VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_profile_attr UNIQUE(user_id, category, attribute)
);

-- Tabela de Memórias Episódicas & Linha Autobiográfica Compartilhada
CREATE TABLE IF NOT EXISTS episodic_memories (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL DEFAULT 'local:owner',
    title VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    emotional_tone VARCHAR(50) DEFAULT 'neutro',
    key_topics JSONB DEFAULT '[]'::jsonb,
    occurred_on DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Índices para busca rápida de perfil e memórias episódicas
CREATE INDEX IF NOT EXISTS idx_user_profiles_lookup ON user_profiles(user_id, category);
CREATE INDEX IF NOT EXISTS idx_episodic_memories_date ON episodic_memories(user_id, occurred_on DESC);

-- Tabela de estado do Worker de Memória e Consolidação
CREATE TABLE IF NOT EXISTS memory_worker_state (
    id VARCHAR(100) PRIMARY KEY,
    last_processed_id INT DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Índice HNSW de alta performance para busca por similaridade de cosseno
CREATE INDEX IF NOT EXISTS memories_embedding_idx 
ON memories 
USING hnsw (embedding vector_cosine_ops);
