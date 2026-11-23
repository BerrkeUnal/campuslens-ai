
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, OpenAIError

from backend.app.rag.llm_client import LLMClient, LLMClientError


@pytest.mark.asyncio
async def test_llm_returns_structured_json():
    client = LLMClient.__new__(LLMClient)
    client.model = "gpt-4.1-mini"
    client.temperature = 0

    mock_response = SimpleNamespace(
        status="completed",
        output_text=(
            '{"answer": "Bütünleme sınavına giremezsiniz. [S1]", '
            '"used_source_ids": ["S1"], '
            '"insufficient_context": false}'
        ),
    )

    client.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(return_value=mock_response)
        )
    )

    result = await client.generate("Mock university question")

    assert result.used_source_ids == ["S1"]
    assert result.insufficient_context is False
    assert "[S1]" in result.answer

    client.client.responses.create.assert_awaited_once()

    kwargs = client.client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1-mini"
    assert kwargs["temperature"] == 0
    assert kwargs["max_output_tokens"] == 400


@pytest.mark.asyncio
async def test_llm_rejects_invalid_json():
    client = LLMClient.__new__(LLMClient)
    client.model = "gpt-4.1-mini"
    client.temperature = 0

    client.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    status="completed",
                    output_text="not valid JSON",
                )
            )
        )
    )

    with pytest.raises(LLMClientError):
        await client.generate("Mock question")


@pytest.mark.asyncio
async def test_llm_retries_once_after_invalid_json():
    client = LLMClient.__new__(LLMClient)
    client.model = "gpt-4.1-mini"
    client.temperature = 0

    invalid_response = SimpleNamespace(
        status="completed",
        output_text="not valid JSON",
    )

    valid_response = SimpleNamespace(
        status="completed",
        output_text=(
            '{"answer": "Bütünlemeye giremezsiniz. [S1]", '
            '"used_source_ids": ["S1"], '
            '"insufficient_context": false}'
        ),
    )

    mock_create = AsyncMock(
        side_effect=[invalid_response, valid_response]
    )

    client.client = SimpleNamespace(
        responses=SimpleNamespace(create=mock_create)
    )

    result = await client.generate("Mock question")

    assert result.used_source_ids == ["S1"]
    assert result.insufficient_context is False
    assert mock_create.await_count == 2


@pytest.mark.asyncio
async def test_llm_fails_after_two_invalid_responses():
    client = LLMClient.__new__(LLMClient)
    client.model = "gpt-4.1-mini"
    client.temperature = 0

    invalid_response = SimpleNamespace(
        status="completed",
        output_text="not valid JSON",
    )

    mock_create = AsyncMock(return_value=invalid_response)

    client.client = SimpleNamespace(
        responses=SimpleNamespace(create=mock_create)
    )

    with pytest.raises(
        LLMClientError,
        match="iki denemede de",
    ):
        await client.generate("Mock question")

    assert mock_create.await_count == 2

def test_llm_rejects_missing_api_key():
    settings = SimpleNamespace(
        llm_model="gpt-4.1-mini",
        llm_temperature=0,
        openai_api_key="",
    )

    with patch(
        "backend.app.rag.llm_client.get_settings",
        return_value=settings,
    ), patch(
        "backend.app.rag.llm_client.AsyncOpenAI"
    ) as mock_openai:
        with pytest.raises(
            LLMClientError,
            match="API anahtarı yapılandırılmamış",
        ):
            LLMClient()

        mock_openai.assert_not_called()


def test_llm_rejects_whitespace_api_key():
    settings = SimpleNamespace(
        llm_model="gpt-4.1-mini",
        llm_temperature=0,
        openai_api_key="   ",
    )

    with patch(
        "backend.app.rag.llm_client.get_settings",
        return_value=settings,
    ), patch(
        "backend.app.rag.llm_client.AsyncOpenAI"
    ) as mock_openai:
        with pytest.raises(
            LLMClientError,
            match="API anahtarı yapılandırılmamış",
        ):
            LLMClient()

        mock_openai.assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "expected_message"),
    [
        (
            AuthenticationError,
            "LLM servisi için kimlik doğrulama başarısız oldu.",
        ),
        (
            BadRequestError,
            "LLM isteği işlenemedi.",
        ),
    ],
)
async def test_llm_handles_openai_api_errors(
    error_type,
    expected_message,
):
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )

    response = httpx.Response(
        status_code=401 if error_type is AuthenticationError else 400,
        request=request,
        json={"error": {"message": "SECRET_TECHNICAL_DETAIL"}},
    )

    api_error = error_type(
        "SECRET_TECHNICAL_DETAIL",
        response=response,
        body=None,
    )

    client = LLMClient.__new__(LLMClient)
    client.model = "gpt-4.1-mini"
    client.temperature = 0

    mock_create = AsyncMock(side_effect=api_error)

    client.client = SimpleNamespace(
        responses=SimpleNamespace(create=mock_create)
    )

    with pytest.raises(LLMClientError) as exc_info:
        await client.generate("Mock question")

    assert str(exc_info.value) == expected_message
    assert "SECRET_TECHNICAL_DETAIL" not in str(exc_info.value)
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_handles_unexpected_openai_error():
    client = LLMClient.__new__(LLMClient)
    client.model = "gpt-4.1-mini"
    client.temperature = 0

    mock_create = AsyncMock(
        side_effect=OpenAIError("SECRET_TECHNICAL_DETAIL")
    )

    client.client = SimpleNamespace(
        responses=SimpleNamespace(create=mock_create)
    )

    with pytest.raises(
        LLMClientError,
        match="LLM servisinde beklenmeyen bir hata oluştu.",
    ) as exc_info:
        await client.generate("Mock question")

    assert "SECRET_TECHNICAL_DETAIL" not in str(exc_info.value)
    mock_create.assert_awaited_once()