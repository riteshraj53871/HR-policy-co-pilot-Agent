from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from app.config import get_settings


@lru_cache
def _index():
    s = get_settings()
    pc = Pinecone(api_key=s.pinecone_api_key)
    return pc.Index(s.pinecone_index)


def ensure_index() -> None:
    """Create the index if it doesn't exist (safe to call repeatedly)."""
    s = get_settings()
    pc = Pinecone(api_key=s.pinecone_api_key)
    if s.pinecone_index not in pc.list_indexes().names():
        pc.create_index(
            name=s.pinecone_index,
            dimension=s.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        )


def upsert_chunks(chunks: list[dict], batch_size: int = 100) -> int:
    """chunks: [{"id": str, "values": list[float], "metadata": dict}]
    The chunk text must be stored in metadata["text"]."""
    ns = get_settings().pinecone_namespace
    for i in range(0, len(chunks), batch_size):
        _index().upsert(vectors=chunks[i : i + batch_size], namespace=ns)
    return len(chunks)


def query(vector: list[float], top_k: int | None = None, filter: dict | None = None) -> list[dict]:
    """Returns [{"id", "score", "text", "metadata"}], best match first."""
    s = get_settings()
    res = _index().query(
        vector=vector,
        top_k=top_k or s.top_k,
        namespace=s.pinecone_namespace,
        filter=filter,
        include_metadata=True,
    )
    return [
        {
            "id": m.id,
            "score": m.score,
            "text": (m.metadata or {}).get("text", ""),
            "metadata": m.metadata or {},
        }
        for m in res.matches
    ]


def stats() -> dict:
    return _index().describe_index_stats().to_dict()

def reset_namespace() -> None:
    _index().delete(delete_all=True, namespace=get_settings().pinecone_namespace)