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
        ollama_model="qwen3:4b",
        ollama_keep_alive="30m",
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
    assert payload["model"] == "qwen3:4b"
    assert payload["keep_alive"] == "30m"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert "format" in payload
    messages = payload["messages"]

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"].startswith("/no_think\n")
    assert messages[-1]["content"].endswith("Test kullanıcı promptu")

    few_shot_outputs = [
        json.loads(message["content"])
        for message in messages
        if message["role"] == "assistant"
    ]

    assert len(few_shot_outputs) == 2
    assert few_shot_outputs[0]["insufficient_context"] is False
    assert few_shot_outputs[1]["insufficient_context"] is True


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

    first_payload = mock_request.await_args_list[0].args[0]
    second_payload = mock_request.await_args_list[1].args[0]

    assert first_payload["keep_alive"] == "30m"
    assert second_payload["keep_alive"] == "30m"

    assert first_payload["messages"][-1] != second_payload["messages"][-1]
    assert (
        "Önceki denemede geçerli JSON üretilemedi"
        in second_payload["messages"][-1]["content"]
    )


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


@pytest.mark.asyncio
async def test_ollama_accepts_insufficient_context_output() -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    response_content = json.dumps(
        {
            "answer": "Verilen kaynaklarda bu bilgi bulunmuyor.",
            "used_source_ids": [],
            "insufficient_context": True,
        }
    )
    client._request = AsyncMock(return_value=make_response(200, response_content))

    output = await client.generate("Yurt ücreti ne kadar?")

    assert output.insufficient_context is True
    assert output.used_source_ids == []


@pytest.fixture
def availability_client():
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        return OllamaLLMClient()


@pytest.fixture
def availability_http_client():
    with patch("backend.app.rag.ollama_client.httpx.AsyncClient") as factory:
        http_client = factory.return_value.__aenter__.return_value
        http_client.post = AsyncMock()
        yield factory, http_client


def make_availability_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json={},
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:11434/api/show",
        ),
    )


@pytest.mark.asyncio
async def test_ollama_availability_accepts_installed_model(
    availability_client,
    availability_http_client,
) -> None:
    factory, http_client = availability_http_client
    http_client.post.return_value = make_availability_response(200)

    await availability_client.check_model_available()

    factory.assert_called_once_with(timeout=10.0)
    http_client.post.assert_awaited_once_with(
        "http://127.0.0.1:11434/api/show",
        json={"model": "qwen3:4b"},
    )


@pytest.mark.asyncio
async def test_ollama_availability_rejects_missing_model(
    availability_client,
    availability_http_client,
) -> None:
    _, http_client = availability_http_client
    http_client.post.return_value = make_availability_response(404)

    with pytest.raises(LLMClientError) as exc_info:
        await availability_client.check_model_available()

    assert "qwen3:4b" in str(exc_info.value)
    assert "scripts.prepare_models" in str(exc_info.value)
    assert exc_info.value.status_code == 503

    http_client.post.assert_awaited_once_with(
        "http://127.0.0.1:11434/api/show",
        json={"model": "qwen3:4b"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "expected_message"),
    [
        (
            httpx.ConnectError,
            "Ollama servisine bağlanılamadı",
        ),
        (
            httpx.ReadTimeout,
            "Ollama model kontrolü zamanında tamamlanamadı",
        ),
        (
            httpx.ReadError,
            "Ollama servisiyle iletişim kurulamadı",
        ),
    ],
)
async def test_ollama_availability_handles_request_errors(
    availability_client,
    availability_http_client,
    error_type,
    expected_message,
) -> None:
    _, http_client = availability_http_client

    http_client.post.side_effect = error_type(
        "Simulated request failure",
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:11434/api/show",
        ),
    )

    with pytest.raises(
        LLMClientError,
        match=expected_message,
    ) as exc_info:
        await availability_client.check_model_available()

    assert exc_info.value.status_code == 503
    http_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_ollama_availability_handles_server_error(
    availability_client,
    availability_http_client,
) -> None:
    _, http_client = availability_http_client
    http_client.post.return_value = make_availability_response(500)

    with pytest.raises(
        LLMClientError,
        match="Ollama model kontrolü başarısız oldu",
    ) as exc_info:
        await availability_client.check_model_available()

    assert exc_info.value.status_code == 503
    http_client.post.assert_awaited_once()


def make_warmup_response(
    status_code: int = 200,
    content: str = '{"status":"ready"}',
    **overrides,
) -> httpx.Response:
    data = {
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "done_reason": "stop",
        "eval_count": 10,
    }
    data.update(overrides)

    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:11434/api/chat",
        ),
    )


@pytest.mark.asyncio
async def test_ollama_warm_up_accepts_valid_technical_output(
    availability_client,
) -> None:
    client = availability_client
    client.check_model_available = AsyncMock()
    client._request = AsyncMock(
        return_value=make_warmup_response()
    )

    await client.warm_up()

    client.check_model_available.assert_awaited_once_with()
    client._request.assert_awaited_once()

    payload = client._request.await_args.args[0]

    assert payload["model"] == "qwen3:4b"
    assert payload["keep_alive"] == "30m"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["num_predict"] == 64

    schema = payload["format"]
    assert schema["properties"]["status"]["type"] == "string"
    assert schema["required"] == ["status"]

    messages = payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("/no_think")
    assert '{"status":"ready"}' in messages[0]["content"]


@pytest.mark.asyncio
async def test_ollama_warm_up_skips_generation_if_model_unavailable(
    availability_client,
) -> None:
    client = availability_client

    client.check_model_available = AsyncMock(
        side_effect=LLMClientError("Model unavailable.")
    )
    client._request = AsyncMock()

    with pytest.raises(LLMClientError, match="Model unavailable"):
        await client.warm_up()

    client.check_model_available.assert_awaited_once_with()
    client._request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "invalid-json",
        "",
        "{}",
        '{"status":null}',
        '{"status":123}',
        '{"answer":"ready"}',
    ],
)
async def test_ollama_warm_up_rejects_invalid_output(
    availability_client,
    content,
) -> None:
    client = availability_client
    client.check_model_available = AsyncMock()
    client._request = AsyncMock(
        return_value=make_warmup_response(content=content)
    )

    with pytest.raises(
        LLMClientError,
        match="Ollama hazırlık isteği geçersiz çıktı üretti",
    ) as exc_info:
        await client.warm_up()

    assert exc_info.value.status_code == 503
    client.check_model_available.assert_awaited_once_with()
    client._request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"status":"not_ready"}',
        '{"status":"READY"}',
        '{"status":""}',
    ],
)
async def test_ollama_warm_up_rejects_incorrect_status(
    availability_client,
    content,
) -> None:
    client = availability_client
    client.check_model_available = AsyncMock()
    client._request = AsyncMock(
        return_value=make_warmup_response(content=content)
    )

    with pytest.raises(
        LLMClientError,
        match="Ollama hazırlık isteği tamamlanamadı",
    ) as exc_info:
        await client.warm_up()

    assert exc_info.value.status_code == 503
    client._request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"done": False},
        {"done": None},
        {"done": 1},
        {"done_reason": "length"},
        {"done_reason": None},
        {"eval_count": 0},
        {"eval_count": -1},
        {"eval_count": None},
        {"eval_count": True},
        {"eval_count": 1.5},
        {"eval_count": "10"},
    ],
)
async def test_ollama_warm_up_rejects_incomplete_inference(
    availability_client,
    overrides,
) -> None:
    client = availability_client
    client.check_model_available = AsyncMock()
    client._request = AsyncMock(
        return_value=make_warmup_response(**overrides)
    )

    with pytest.raises(
        LLMClientError,
        match="Ollama hazırlık isteği tamamlanamadı",
    ) as exc_info:
        await client.warm_up()

    assert exc_info.value.status_code == 503
    client._request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_field",
    ["message", "done", "done_reason", "eval_count"],
)
async def test_ollama_warm_up_rejects_missing_response_fields(
    availability_client,
    missing_field,
) -> None:
    client = availability_client
    client.check_model_available = AsyncMock()

    data = make_warmup_response().json()
    del data[missing_field]

    response = httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:11434/api/chat",
        ),
    )

    client._request = AsyncMock(return_value=response)

    expected_message = (
        "Ollama hazırlık isteği geçersiz çıktı üretti"
        if missing_field == "message"
        else "Ollama hazırlık isteği tamamlanamadı"
    )

    with pytest.raises(
        LLMClientError,
        match=expected_message,
    ) as exc_info:
        await client.warm_up()

    assert exc_info.value.status_code == 503
    client._request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "expected_message"),
    [
        (
            httpx.ReadTimeout,
            "Ollama hazırlık isteği zamanında tamamlanamadı",
        ),
        (
            httpx.ConnectTimeout,
            "Ollama hazırlık isteği zamanında tamamlanamadı",
        ),
        (
            httpx.ConnectError,
            "Ollama hazırlık isteği başarısız oldu",
        ),
        (
            httpx.ReadError,
            "Ollama hazırlık isteği başarısız oldu",
        ),
    ],
)
async def test_ollama_warm_up_handles_request_errors(
    availability_client,
    error_type,
    expected_message,
) -> None:
    client = availability_client
    client.check_model_available = AsyncMock()

    client._request = AsyncMock(
        side_effect=error_type(
            "Simulated inference failure",
            request=httpx.Request(
                "POST",
                "http://127.0.0.1:11434/api/chat",
            ),
        )
    )

    with pytest.raises(
        LLMClientError,
        match=expected_message,
    ) as exc_info:
        await client.warm_up()

    assert exc_info.value.status_code == 503
    client.check_model_available.assert_awaited_once_with()
    client._request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 500, 503])
async def test_ollama_warm_up_handles_http_errors(
    availability_client,
    status_code,
) -> None:
    client = availability_client
    client.check_model_available = AsyncMock()
    client._request = AsyncMock(
        return_value=make_warmup_response(
            status_code=status_code
        )
    )

    with pytest.raises(
        LLMClientError,
        match="Ollama hazırlık isteği başarısız oldu",
    ) as exc_info:
        await client.warm_up()

    assert exc_info.value.status_code == 503
    client._request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("blank_answer", ["", "   ", "\t\n"])
async def test_ollama_retries_blank_answer_and_recovers(
    blank_answer: str,
) -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    invalid_content = json.dumps(
        {
            "answer": blank_answer,
            "used_source_ids": [],
            "insufficient_context": True,
        }
    )

    valid_content = json.dumps(
        {
            "answer": "Bu soruyu yanıtlamak için yeterli kaynak bulunamadı.",
            "used_source_ids": [],
            "insufficient_context": True,
        }
    )

    client._request = AsyncMock(
        side_effect=[
            make_response(200, invalid_content),
            make_response(200, valid_content),
        ]
    )

    result = await client.generate("Öğrenci yurt ücreti nedir?")

    assert result.answer.strip()
    assert result.insufficient_context is True
    assert result.used_source_ids == []
    assert client._request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("blank_answer", ["", "   ", "\t\n"])
async def test_ollama_rejects_two_blank_answers(
    blank_answer: str,
) -> None:
    with patch(
        "backend.app.rag.ollama_client.get_settings",
        return_value=make_settings(),
    ):
        client = OllamaLLMClient()

    invalid_content = json.dumps(
        {
            "answer": blank_answer,
            "used_source_ids": [],
            "insufficient_context": True,
        }
    )

    client._request = AsyncMock(
        side_effect=[
            make_response(200, invalid_content),
            make_response(200, invalid_content),
        ]
    )

    with pytest.raises(
        LLMClientError,
        match="iki denemede de geçerli JSON",
    ):
        await client.generate("Öğrenci yurt ücreti nedir?")

    assert client._request.await_count == 2
