"""Verify CUDA failures propagate through model readiness."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.app.core.config import Settings
from backend.app.rag.model_readiness import (
    ModelReadinessManager,
    ReadinessState,
)
from backend.app.rag.model_readiness_guard import (
    ModelUnavailableError,
    ensure_models_ready,
)
from backend.app.rag.reranker import Reranker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_mode",
    [
        "cuda_unavailable",
        "zero_cuda_devices",
        "model_loading_failure",
    ],
)
async def test_cuda_failure_propagates_to_readiness(
    failure_mode: str,
) -> None:
    module = "backend.app.rag.reranker"

    settings = Settings(
        _env_file=None,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_device="cuda:0",
    )

    embedding = Mock()
    llm = AsyncMock()

    with (
        patch(
            f"{module}.get_settings",
            return_value=settings,
        ),
        patch(
            f"{module}.torch.cuda.is_available",
            return_value=failure_mode != "cuda_unavailable",
        ) as available,
        patch(
            f"{module}.torch.cuda.device_count",
            return_value=(
                1 if failure_mode == "model_loading_failure" else 0
            ),
        ) as count,
        patch(f"{module}.CrossEncoder") as encoder,
    ):
        if failure_mode == "model_loading_failure":
            encoder.side_effect = RuntimeError(
                "Simulated CUDA model loading failure"
            )

        reranker = Reranker()

        manager = ModelReadinessManager(
            embedding_warmup=embedding,
            reranker_warmup=reranker.warm_up,
            llm_warmup=llm,
        )

        assert manager.snapshot().state == ReadinessState.NOT_STARTED

        await manager.warm_up()

        snapshot = manager.snapshot()

        assert snapshot.state == ReadinessState.FAILED
        assert snapshot.failed_component == "reranker"
        assert snapshot.completed_components == ("embedding",)
        assert snapshot.active_component is None

        durations = dict(snapshot.component_durations_ms)

        assert set(durations) == {"embedding", "reranker"}
        assert all(value >= 0 for value in durations.values())
        assert snapshot.total_duration_ms is not None
        assert snapshot.total_duration_ms >= 0

        embedding.assert_called_once_with()
        llm.assert_not_called()
        llm.assert_not_awaited()

        available.assert_called_once_with()

        if failure_mode == "cuda_unavailable":
            count.assert_not_called()
        else:
            count.assert_called_once_with()

        if failure_mode == "model_loading_failure":
            encoder.assert_called_once_with(
                "BAAI/bge-reranker-v2-m3",
                device="cuda:0",
                local_files_only=True,
            )
        else:
            encoder.assert_not_called()

        encoder.return_value.predict.assert_not_called()

        with pytest.raises(ModelUnavailableError) as error:
            ensure_models_ready(manager)

        assert error.value.status_code == 503
        assert error.value.code == "model_unavailable"

        previous_calls = (
            embedding.call_count,
            available.call_count,
            count.call_count,
            encoder.call_count,
            llm.call_count,
        )

        await manager.warm_up()

        assert manager.snapshot() == snapshot

        assert (
            embedding.call_count,
            available.call_count,
            count.call_count,
            encoder.call_count,
            llm.call_count,
        ) == previous_calls
