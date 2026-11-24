from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Standard API prices in USD per one million tokens.
# Reference: https://openai.com/api/pricing/
LLM_MODEL_PRICES_USD_PER_1M_TOKENS = {
    "gpt-4.1-mini": {
        "input": 0.40,
        "output": 1.60,
    },
}


class Settings(BaseSettings):
    app_env: str = "development"
    frontend_origin: str = "http://127.0.0.1:5500"

    database_url: str = "sqlite:///./data/campuslens.db"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "campuslens_chunks"

    embedding_model: str = "intfloat/multilingual-e5-small"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0

    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 60

    dense_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 5
    max_upload_mb: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
