from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.app.rag.llm_client import LLMClientError
from backend.app.rag.llm_provider import create_llm_client


def test_provider_creates_ollama_client() -> None:
    fake_client = object()
    settings = SimpleNamespace(llm_provider=" OLLAMA ")

    with (
        patch(
            "backend.app.rag.llm_provider.get_settings",
            return_value=settings,
        ),
        patch(
            "backend.app.rag.ollama_client.OllamaLLMClient",
            return_value=fake_client,
        ) as mock_ollama_client,
    ):
        result = create_llm_client()

    assert result is fake_client
    mock_ollama_client.assert_called_once_with()


def test_provider_creates_openai_client() -> None:
    fake_client = object()
    settings = SimpleNamespace(llm_provider="openai")

    with (
        patch(
            "backend.app.rag.llm_provider.get_settings",
            return_value=settings,
        ),
        patch(
            "backend.app.rag.llm_provider.LLMClient",
            return_value=fake_client,
        ) as mock_openai_client,
    ):
        result = create_llm_client()

    assert result is fake_client
    mock_openai_client.assert_called_once_with()


def test_provider_rejects_unknown_provider() -> None:
    settings = SimpleNamespace(llm_provider="unknown")

    with patch(
        "backend.app.rag.llm_provider.get_settings",
        return_value=settings,
    ), pytest.raises(
        LLMClientError,
        match="Desteklenmeyen LLM sağlayıcısı",
    ):
        create_llm_client()


def test_provider_rejects_empty_provider() -> None:
    settings = SimpleNamespace(llm_provider="   ")

    with patch(
        "backend.app.rag.llm_provider.get_settings",
        return_value=settings,
    ), pytest.raises(
        LLMClientError,
        match="boş",
    ):
        create_llm_client()
