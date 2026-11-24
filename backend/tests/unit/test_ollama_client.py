import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.rag.llm_client import LLMClientError
from backend.app.rag.ollama_client import OllamaLLMClient


def make_settings():
    return SimpleNamespace(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
        llm_temperature=0,
    )


def make_response(
    status_code: int,
    content: str,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json={
            "message": {
                "role": "assistant",
                "content": content,
            }
        },
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:11434/api/chat",
        ),
    )


@pytest.mark.asyncio
async def test_ollama_generates_structured_output() -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    response_content = json.dumps(
        {
            "answer": "Öğrenci bütünleme sınavına katılabilir. [S1]",
            "used_source_ids": ["S1"],
            "insufficient_context": False,
        }
    )
    mock_request = AsyncMock(return_value=make_response(200, response_content))
    client._request = mock_request

    output = await client.generate("Test kullanıcı promptu")

    assert output.answer.endswith("[S1]")
    assert output.used_source_ids == ["S1"]
    assert output.insufficient_context is False

    payload = mock_request.await_args.args[0]
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert "format" in payload
    assert payload["messages"][1]["content"] == "Test kullanıcı promptu"


@pytest.mark.asyncio
async def test_ollama_retries_invalid_json_once() -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    valid_content = json.dumps(
        {
            "answer": "Geçerli yanıt. [S1]",
            "used_source_ids": ["S1"],
            "insufficient_context": False,
        }
    )
    mock_request = AsyncMock(
        side_effect=[
            make_response(200, "geçersiz-json"),
            make_response(200, valid_content),
        ]
    )
    client._request = mock_request

    output = await client.generate("Test promptu")

    assert output.answer == "Geçerli yanıt. [S1]"
    assert mock_request.await_count == 2


@pytest.mark.asyncio
async def test_ollama_reports_connection_error() -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    request = httpx.Request(
        "POST",
        "http://127.0.0.1:11434/api/chat",
    )
    client._request = AsyncMock(
        side_effect=httpx.ConnectError(
            "Connection failed",
            request=request,
        )
    )

    with pytest.raises(
        LLMClientError,
        match="Ollama servisine bağlanılamadı",
    ):
        await client.generate("Test promptu")


@pytest.mark.asyncio
async def test_ollama_reports_missing_model() -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    client._request = AsyncMock(
        return_value=make_response(
            404,
            '{"error": "model not found"}',
        )
    )

    with pytest.raises(
        LLMClientError,
        match="Ollama modeli bulunamadı",
    ):
        await client.generate("Test promptu")


@pytest.mark.asyncio
async def test_ollama_rejects_invalid_output_twice() -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    client._request = AsyncMock(
        side_effect=[
            make_response(200, "invalid"),
            make_response(200, "still-invalid"),
        ]
    )

    with pytest.raises(
        LLMClientError,
        match="iki denemede de geçerli JSON",
    ):
        await client.generate("Test promptu")
