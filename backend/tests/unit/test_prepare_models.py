from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from scripts import prepare_models


@pytest.fixture
def settings(monkeypatch):
    config = SimpleNamespace(
        llm_provider="ollama",
        ollama_model="qwen3:4b",
        ollama_base_url="http://127.0.0.1:11434",
    )

    monkeypatch.setattr(
        prepare_models,
        "get_settings",
        lambda: config,
    )

    return config


def mock_client(monkeypatch, responses):
    client = AsyncMock()
    client.post.side_effect = responses

    context = AsyncMock()
    context.__aenter__.return_value = client

    monkeypatch.setattr(
        prepare_models.httpx,
        "AsyncClient",
        Mock(return_value=context),
    )

    return client


def response(status_code):
    return httpx.Response(
        status_code,
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:11434",
        ),
    )


@pytest.mark.asyncio
async def test_existing_ollama_model_is_not_downloaded(monkeypatch, settings):
    client = mock_client(
        monkeypatch,
        [response(200)],
    )

    await prepare_models.prepare_ollama()

    client.post.assert_awaited_once_with(
        "http://127.0.0.1:11434/api/show",
        json={"model": "qwen3:4b"},
    )


@pytest.mark.asyncio
async def test_missing_ollama_model_is_downloaded(monkeypatch, settings):
    client = mock_client(
        monkeypatch,
        [
            response(404),
            response(200),
            response(200),
        ],
    )

    await prepare_models.prepare_ollama()

    assert client.post.await_count == 3

    calls = client.post.await_args_list

    assert calls[0].args[0].endswith("/api/show")
    assert calls[1].args[0].endswith("/api/pull")
    assert calls[2].args[0].endswith("/api/show")

    assert calls[1].kwargs["json"] == {
        "model": "qwen3:4b",
        "stream": False,
    }


@pytest.mark.asyncio
async def test_ollama_preparation_skipped_for_openai(monkeypatch, settings):
    settings.llm_provider = "openai"

    factory = Mock()

    monkeypatch.setattr(
        prepare_models.httpx,
        "AsyncClient",
        factory,
    )

    await prepare_models.prepare_ollama()

    factory.assert_not_called()


@pytest.mark.asyncio
async def test_ollama_connection_failure(monkeypatch, settings):
    client = mock_client(
        monkeypatch,
        [httpx.ConnectError("Connection refused")],
    )

    with pytest.raises(httpx.ConnectError):
        await prepare_models.prepare_ollama()

    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_ollama_download_failure(monkeypatch, settings):
    client = mock_client(
        monkeypatch,
        [response(404), response(500)],
    )

    with pytest.raises(httpx.HTTPStatusError):
        await prepare_models.prepare_ollama()

    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_ollama_model_still_missing_after_download(monkeypatch, settings):
    client = mock_client(
        monkeypatch,
        [response(404), response(200), response(404)],
    )

    with pytest.raises(httpx.HTTPStatusError):
        await prepare_models.prepare_ollama()

    assert client.post.await_count == 3


def test_main_returns_failure_exit_code(monkeypatch):
    monkeypatch.setattr(
        prepare_models,
        "prepare_models",
        AsyncMock(side_effect=RuntimeError("Preparation failed")),
    )

    assert prepare_models.main() == 1


def test_prepare_embeddings_loads_models(monkeypatch):
    config = SimpleNamespace(
        embedding_model="intfloat/multilingual-e5-small",
        sparse_model="Qdrant/bm25",
    )

    monkeypatch.setattr(
        prepare_models,
        "get_settings",
        lambda: config,
    )

    embedder = Mock()
    factory = Mock(return_value=embedder)

    monkeypatch.setattr(
        prepare_models,
        "TextEmbedder",
        factory,
    )

    prepare_models.prepare_embeddings()

    factory.assert_called_once_with(
        "intfloat/multilingual-e5-small",
        "Qdrant/bm25",
        local_files_only=False,
    )

    embedder.load.assert_called_once_with()


def test_prepare_reranker_loads_configured_model(monkeypatch):
    config = SimpleNamespace(
        reranker_model="BAAI/bge-reranker-v2-m3",
    )

    monkeypatch.setattr(
        prepare_models,
        "get_settings",
        lambda: config,
    )

    factory = Mock()

    monkeypatch.setattr(
        prepare_models,
        "CrossEncoder",
        factory,
    )

    prepare_models.prepare_reranker()

    factory.assert_called_once_with(
        "BAAI/bge-reranker-v2-m3",
        device="cpu",
        local_files_only=False,
    )


@pytest.mark.asyncio
async def test_prepare_models_runs_in_order(monkeypatch):
    calls = []

    def embeddings():
        calls.append("embedding")

    def reranker():
        calls.append("reranker")

    async def ollama():
        calls.append("ollama")

    monkeypatch.setattr(prepare_models, "prepare_embeddings", embeddings)
    monkeypatch.setattr(prepare_models, "prepare_reranker", reranker)
    monkeypatch.setattr(prepare_models, "prepare_ollama", ollama)

    await prepare_models.prepare_models()

    assert calls == ["embedding", "reranker", "ollama"]


@pytest.mark.asyncio
async def test_preparation_stops_on_embedding_failure(monkeypatch):
    embeddings = Mock(side_effect=RuntimeError("Embedding unavailable"))
    reranker = Mock()
    ollama = AsyncMock()

    monkeypatch.setattr(prepare_models, "prepare_embeddings", embeddings)
    monkeypatch.setattr(prepare_models, "prepare_reranker", reranker)
    monkeypatch.setattr(prepare_models, "prepare_ollama", ollama)

    with pytest.raises(RuntimeError, match="Embedding unavailable"):
        await prepare_models.prepare_models()

    reranker.assert_not_called()
    ollama.assert_not_called()


def test_main_returns_success_exit_code(monkeypatch):
    preparation = AsyncMock()

    monkeypatch.setattr(
        prepare_models,
        "prepare_models",
        preparation,
    )

    assert prepare_models.main() == 0
    preparation.assert_awaited_once_with()
