"""Verify benchmark reports reflect configured reranker devices."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.app.core.config import Settings
from scripts.benchmark_model_warmup import run_benchmark


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_benchmark_reports_configured_reranker_device(
    device: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    settings = Settings(
        _env_file=None,
        reranker_device=device,
        llm_provider="ollama",
        ollama_model="qwen3:4b",
        ollama_keep_alive="30m",
    )

    embedder = Mock()
    reranker = Mock()
    llm = Mock()

    llm.warm_up = AsyncMock()
    llm.aclose = AsyncMock()

    module = "scripts.benchmark_model_warmup"
    output = tmp_path / "benchmark.json"

    with (
        patch(
            f"{module}.get_settings",
            return_value=settings,
        ),
        patch(
            f"{module}.get_embedder",
            return_value=embedder,
        ),
        patch(
            f"{module}.get_reranker",
            return_value=reranker,
        ),
        patch(
            f"{module}.OllamaLLMClient",
            return_value=llm,
        ),
    ):
        await run_benchmark(
            repeats=1,
            output=output,
        )

    assert output.is_file()

    report = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert report["models"]["reranker_device"] == device
    assert report["models"]["reranker"] == settings.reranker_model
    assert report["models"]["dense"] == settings.embedding_model
    assert report["models"]["sparse"] == settings.sparse_model
    assert report["models"]["llm"] == "qwen3:4b"
    assert report["models"]["ollama_keep_alive"] == "30m"

    assert report["warm_repeats"] == 1
    assert report["offline_model_loading"] is True

    assert set(report["results"]) == {
        "embedding",
        "reranker",
        "llm",
    }

    for result in report["results"].values():
        assert result["cold_ms"] >= 0
        assert len(result["warm_ms"]) == 1
        assert result["warm_ms"][0] >= 0

    assert embedder.warm_up.call_count == 2
    assert reranker.warm_up.call_count == 2

    assert llm.warm_up.await_count == 2
    llm.aclose.assert_awaited_once_with()
