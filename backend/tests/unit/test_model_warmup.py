import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.app.rag.llm_client import LLMClientError
from backend.app.rag.model_readiness import ReadinessState
from backend.app.rag.model_warmup import (
    create_model_readiness_manager,
    warm_up_embeddings,
    warm_up_ollama,
    warm_up_reranker,
)

MODULE = "backend.app.rag.model_warmup"


def test_embedding_warmup_uses_shared_embedder() -> None:
    embedder = Mock()

    with patch(
        f"{MODULE}.get_embedder",
        return_value=embedder,
    ) as factory:
        warm_up_embeddings()

    factory.assert_called_once_with()
    embedder.warm_up.assert_called_once_with()


def test_reranker_warmup_uses_shared_reranker() -> None:
    reranker = Mock()

    with patch(
        f"{MODULE}.get_reranker",
        return_value=reranker,
    ) as factory:
        warm_up_reranker()

    factory.assert_called_once_with()
    reranker.warm_up.assert_called_once_with()


@pytest.mark.asyncio
async def test_ollama_warmup_closes_client() -> None:
    client = Mock()
    client.warm_up = AsyncMock()
    client.aclose = AsyncMock()

    with patch(
        f"{MODULE}.OllamaLLMClient",
        return_value=client,
    ) as factory:
        await warm_up_ollama()

    factory.assert_called_once_with()
    client.warm_up.assert_awaited_once_with()
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_ollama_failure_still_closes_client() -> None:
    client = Mock()
    client.warm_up = AsyncMock(
        side_effect=LLMClientError("Simulated warm-up failure")
    )
    client.aclose = AsyncMock()

    with (
        patch(
            f"{MODULE}.OllamaLLMClient",
            return_value=client,
        ),
        pytest.raises(LLMClientError),
    ):
        await warm_up_ollama()

    client.warm_up.assert_awaited_once_with()
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_ollama_cancellation_still_closes_client() -> None:
    client = Mock()
    client.warm_up = AsyncMock(
        side_effect=asyncio.CancelledError
    )
    client.aclose = AsyncMock()

    with (
        patch(
            f"{MODULE}.OllamaLLMClient",
            return_value=client,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await warm_up_ollama()

    client.warm_up.assert_awaited_once_with()
    client.aclose.assert_awaited_once_with()


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_factory_does_not_load_models(provider: str) -> None:
    settings = SimpleNamespace(llm_provider=provider)

    with (
        patch(
            f"{MODULE}.get_settings",
            return_value=settings,
        ),
        patch(f"{MODULE}.get_embedder") as embedder_factory,
        patch(f"{MODULE}.get_reranker") as reranker_factory,
        patch(f"{MODULE}.OllamaLLMClient") as ollama_factory,
    ):
        manager = create_model_readiness_manager()

    assert manager.snapshot().state == ReadinessState.NOT_STARTED

    embedder_factory.assert_not_called()
    reranker_factory.assert_not_called()
    ollama_factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", " OLLAMA "])
async def test_ollama_provider_runs_all_components(
    provider: str,
) -> None:
    settings = SimpleNamespace(llm_provider=provider)

    embedder = Mock()
    reranker = Mock()
    client = Mock()
    client.warm_up = AsyncMock()
    client.aclose = AsyncMock()

    with (
        patch(
            f"{MODULE}.get_settings",
            return_value=settings,
        ),
        patch(
            f"{MODULE}.get_embedder",
            return_value=embedder,
        ) as embedder_factory,
        patch(
            f"{MODULE}.get_reranker",
            return_value=reranker,
        ) as reranker_factory,
        patch(
            f"{MODULE}.OllamaLLMClient",
            return_value=client,
        ) as ollama_factory,
    ):
        manager = create_model_readiness_manager()
        await manager.warm_up()

    snapshot = manager.snapshot()

    assert snapshot.state == ReadinessState.READY
    assert snapshot.completed_components == (
        "embedding",
        "reranker",
        "llm",
    )

    embedder_factory.assert_called_once_with()
    reranker_factory.assert_called_once_with()
    ollama_factory.assert_called_once_with()

    embedder.warm_up.assert_called_once_with()
    reranker.warm_up.assert_called_once_with()
    client.warm_up.assert_awaited_once_with()
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", " OPENAI "])
async def test_openai_provider_skips_local_llm(
    provider: str,
) -> None:
    settings = SimpleNamespace(llm_provider=provider)

    embedder = Mock()
    reranker = Mock()

    with (
        patch(
            f"{MODULE}.get_settings",
            return_value=settings,
        ),
        patch(
            f"{MODULE}.get_embedder",
            return_value=embedder,
        ),
        patch(
            f"{MODULE}.get_reranker",
            return_value=reranker,
        ),
        patch(f"{MODULE}.OllamaLLMClient") as ollama_factory,
    ):
        manager = create_model_readiness_manager()
        await manager.warm_up()

    snapshot = manager.snapshot()

    assert snapshot.state == ReadinessState.READY
    assert snapshot.completed_components == (
        "embedding",
        "reranker",
    )

    embedder.warm_up.assert_called_once_with()
    reranker.warm_up.assert_called_once_with()
    ollama_factory.assert_not_called()


@pytest.mark.parametrize("provider", ["unknown", "", "   "])
def test_factory_rejects_invalid_provider(
    provider: str,
) -> None:
    settings = SimpleNamespace(llm_provider=provider)

    with (
        patch(
            f"{MODULE}.get_settings",
            return_value=settings,
        ),
        patch(f"{MODULE}.get_embedder") as embedder_factory,
        patch(f"{MODULE}.get_reranker") as reranker_factory,
        patch(f"{MODULE}.OllamaLLMClient") as ollama_factory,
        pytest.raises(
            LLMClientError,
            match="Desteklenmeyen LLM sağlayıcısı",
        ),
    ):
        create_model_readiness_manager()

    embedder_factory.assert_not_called()
    reranker_factory.assert_not_called()
    ollama_factory.assert_not_called()


@pytest.mark.asyncio
async def test_production_manager_records_ollama_failure() -> None:
    settings = SimpleNamespace(llm_provider="ollama")

    embedder = Mock()
    reranker = Mock()
    client = Mock()
    client.warm_up = AsyncMock(
        side_effect=LLMClientError("Simulated Ollama failure")
    )
    client.aclose = AsyncMock()

    with (
        patch(
            f"{MODULE}.get_settings",
            return_value=settings,
        ),
        patch(
            f"{MODULE}.get_embedder",
            return_value=embedder,
        ),
        patch(
            f"{MODULE}.get_reranker",
            return_value=reranker,
        ),
        patch(
            f"{MODULE}.OllamaLLMClient",
            return_value=client,
        ),
    ):
        manager = create_model_readiness_manager()
        await manager.warm_up()

    snapshot = manager.snapshot()

    assert snapshot.state == ReadinessState.FAILED
    assert snapshot.failed_component == "llm"
    assert snapshot.completed_components == (
        "embedding",
        "reranker",
    )
    assert snapshot.active_component is None

    client.warm_up.assert_awaited_once_with()
    client.aclose.assert_awaited_once_with()
