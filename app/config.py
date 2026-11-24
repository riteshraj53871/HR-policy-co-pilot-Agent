from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    groq_api_key: str
    pinecone_api_key: str
    tavily_api_key: str

    # Pinecone
    pinecone_index: str = "hr-copilot"
    pinecone_namespace: str = "hr-policies"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Groq models
    router_model: str = "llama-3.1-8b-instant"
    grader_model: str = "llama-3.1-8b-instant"
    generator_model: str = "llama-3.3-70b-versatile"

    # Agent behavior
    top_k: int = 5
    kb_score_threshold: float = 0.55
    max_retries: int = 2

    # Storage
    audit_db_path: str = "data/audit.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()