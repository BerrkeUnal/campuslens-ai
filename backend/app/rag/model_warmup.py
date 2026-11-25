"""Connect model readiness to the production model factories."""

from backend.app.core.config import get_settings
from backend.app.rag.embedder import get_embedder
from backend.app.rag.llm_client import LLMClientError
from backend.app.rag.model_readiness import ModelReadinessManager
from backend.app.rag.ollama_client import OllamaLLMClient
from backend.app.rag.reranker import get_reranker


def warm_up_embeddings() -> None:
    """Warm up the shared dense and sparse embedding models."""
    get_embedder().warm_up()


def warm_up_reranker() -> None:
    """Warm up the shared cross-encoder model."""
    get_reranker().warm_up()


async def warm_up_ollama() -> None:
    """Verify local LLM inference and release the temporary client."""
    client = OllamaLLMClient()

    try:
        await client.warm_up()
    finally:
        await client.aclose()


def create_model_readiness_manager() -> ModelReadinessManager:
    """Build a readiness manager without loading any models."""
    provider = get_settings().llm_provider.strip().lower()

    if provider == "ollama":
        llm_warmup = warm_up_ollama
    elif provider == "openai":
        llm_warmup = None
    else:
        raise LLMClientError(
            f"Desteklenmeyen LLM sağlayıcısı: {provider or 'boş'}."
        )

    return ModelReadinessManager(
        embedding_warmup=warm_up_embeddings,
        reranker_warmup=warm_up_reranker,
        llm_warmup=llm_warmup,
    )
