from fastembed import TextEmbedding

_model = None


def get_embedding(text: str) -> list[float]:
    """
    Gera um embedding vetorial de 384 dimensões otimizado para múltiplos idiomas (incluindo PT-BR).
    Modelo: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    """
    global _model
    if _model is None:
        _model = TextEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    embedding = next(iter(_model.embed([text])))
    return embedding.tolist()
