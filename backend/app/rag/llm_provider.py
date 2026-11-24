from backend.app.core.config import get_settings
from backend.app.rag.llm_client import (
    LLMClient,
    LLMClientError,
    LLMGenerator,
)


def create_llm_client() -> LLMGenerator:
    """Create the configured LLM provider client."""
    provider = get_settings().llm_provider.strip().lower()

    if provider == "ollama":
        from backend.app.rag.ollama_client import OllamaLLMClient

        return OllamaLLMClient()

    if provider == "openai":
        return LLMClient()

    raise LLMClientError(f"Desteklenmeyen LLM sağlayıcısı: {provider or 'boş'}.")
