from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    # Loaded once per process (first call is slow, later calls are instant)
    return SentenceTransformer(get_settings().embedding_model)


def embed_query(text: str) -> list[float]:
    """Queries need the BGE instruction prefix."""
    s = get_settings()
    vec = _model().encode(s.query_prefix + text, normalize_embeddings=True)
    return vec.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Documents get NO prefix."""
    vecs = _model().encode(texts, normalize_embeddings=True, batch_size=32)
    return vecs.tolist()