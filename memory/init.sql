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

-- Índice HNSW de alta performance para busca por similaridade de cosseno
CREATE INDEX IF NOT EXISTS memories_embedding_idx 
ON memories 
USING hnsw (embedding vector_cosine_ops);
