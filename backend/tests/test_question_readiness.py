"""HTTP integration tests for the question model readiness gate."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.rag.model_readiness import (
    ModelReadinessManager,
    ReadinessState,
)
from backend.app.schemas.questions import (
    MetricsResponse,
    PipelineResult,
)
from backend.app.schemas.retrieval import RetrievalMethod
from backend.app.services.question_service import get_question_pipeline

QUESTION_PATH = "/api/questions/ask"

QUESTION = "Butunleme sinavina kimler girebilir?"

PREPARING_MESSAGE = (
    "Yapay zekâ modelleri hazırlanıyor. "
    "Lütfen kısa bir süre sonra tekrar deneyin."
)

UNAVAILABLE_MESSAGE = (
    "Yapay zekâ modelleri şu anda kullanılamıyor. "
    "Lütfen daha sonra tekrar deneyin."
)

REQUEST_ID = "question-readiness-test"

METHODS = [
    "bm25",
    "dense",
    "hybrid",
    "hybrid_rerank",
]


def build_app(state: ReadinessState | None, tmp_path: Path):
    settings = Settings(
        _env_file=None,
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        model_warmup_enabled=state is not None,
    )

    manager = None

    if state is not None:
        manager = Mock(spec=ModelReadinessManager)

        manager.snapshot.return_value = SimpleNamespace(
            state=state,
        )

    return create_app(
        settings,
        readiness_manager=manager,
    )


def build_pipeline() -> Mock:
    pipeline = Mock()

    pipeline.answer_question = AsyncMock(
        return_value=PipelineResult(
            answer="Test yaniti.",
            insufficient_context=True,
            sources=[],
            metrics=MetricsResponse(
                retrieval_latency_ms=10,
                generation_latency_ms=20,
                total_latency_ms=30,
                estimated_cost_usd=0.0,
                input_tokens=10,
                output_tokens=5,
            ),
        ),
    )

    return pipeline


@pytest.mark.parametrize(
    ("state", "error_code", "message"),
    [
        (
            ReadinessState.NOT_STARTED,
            "model_preparing",
            PREPARING_MESSAGE,
        ),
        (
            ReadinessState.PREPARING,
            "model_preparing",
            PREPARING_MESSAGE,
        ),
        (
            ReadinessState.FAILED,
            "model_unavailable",
            UNAVAILABLE_MESSAGE,
        ),
    ],
)
@pytest.mark.parametrize("method", METHODS)
def test_unready_models_block_all_retrieval_methods(
    state: ReadinessState,
    error_code: str,
    message: str,
    method: str,
    tmp_path: Path,
) -> None:
    app = build_app(state, tmp_path)

    pipeline = build_pipeline()
    factory = Mock(return_value=pipeline)

    app.dependency_overrides[get_question_pipeline] = factory

    client = TestClient(app)

    try:
        response = client.post(
            QUESTION_PATH,
            json={
                "question": QUESTION,
                "method": method,
            },
            headers={"X-Request-ID": REQUEST_ID},
        )

        assert response.status_code == 503

        assert response.json() == {
            "error": error_code,
            "message": message,
            "request_id": REQUEST_ID,
            "details": None,
        }

        assert response.headers["X-Request-ID"] == REQUEST_ID

        factory.assert_not_called()
        pipeline.answer_question.assert_not_awaited()

        app.state.model_readiness.snapshot.assert_called_once_with()
        app.state.model_readiness.warm_up.assert_not_called()

    finally:
        client.close()
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "state",
    [
        ReadinessState.NOT_STARTED,
        ReadinessState.PREPARING,
        ReadinessState.READY,
        ReadinessState.FAILED,
        None,
    ],
)
@pytest.mark.parametrize(
    ("payload", "expected_field"),
    [
        (
            {"question": "a"},
            "body.question",
        ),
        (
            {"question": "   "},
            "body.question",
        ),
        (
            {"question": "x" * 1001},
            "body.question",
        ),
        (
            {"method": "bm25"},
            "body.question",
        ),
        (
            {
                "question": QUESTION,
                "method": "unknown",
            },
            "body.method",
        ),
    ],
)
def test_invalid_requests_are_rejected_before_readiness(
    state: ReadinessState | None,
    payload: dict,
    expected_field: str,
    tmp_path: Path,
) -> None:
    app = build_app(state, tmp_path)

    pipeline = build_pipeline()
    factory = Mock(return_value=pipeline)

    app.dependency_overrides[get_question_pipeline] = factory

    client = TestClient(app)

    try:
        response = client.post(
            QUESTION_PATH,
            json=payload,
            headers={"X-Request-ID": REQUEST_ID},
        )

        assert response.status_code == 422

        body = response.json()

        assert body["error"] == "validation_error"
        assert body["request_id"] == REQUEST_ID
        assert response.headers["X-Request-ID"] == REQUEST_ID

        assert len(body["details"]) == 1
        assert body["details"][0]["field"] == expected_field

        factory.assert_not_called()
        pipeline.answer_question.assert_not_awaited()

        if app.state.model_readiness is not None:
            app.state.model_readiness.snapshot.assert_not_called()

    finally:
        client.close()
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "state",
    [ReadinessState.READY, None],
)
@pytest.mark.parametrize("method", METHODS)
def test_ready_or_disabled_models_allow_execution(
    state: ReadinessState | None,
    method: str,
    tmp_path: Path,
) -> None:
    app = build_app(state, tmp_path)

    pipeline = build_pipeline()
    factory = Mock(return_value=pipeline)

    app.dependency_overrides[get_question_pipeline] = factory

    client = TestClient(app)

    try:
        response = client.post(
            QUESTION_PATH,
            json={
                "question": f"  {QUESTION}  ",
                "method": method,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["method"] == method
        assert body["answer"] == "Test yaniti."
        assert body["insufficient_context"] is True
        assert body["sources"] == []
        assert body["metrics"]["total_latency_ms"] == 30

        factory.assert_called_once_with()

        pipeline.answer_question.assert_awaited_once_with(
            question=QUESTION,
            method=RetrievalMethod(method),
        )

    finally:
        client.close()
        app.dependency_overrides.clear()


def test_async_pipeline_override_is_supported(
    tmp_path: Path,
) -> None:
    app = build_app(ReadinessState.READY, tmp_path)

    pipeline = build_pipeline()
    factory = AsyncMock(return_value=pipeline)

    app.dependency_overrides[get_question_pipeline] = factory

    client = TestClient(app)

    try:
        response = client.post(
            QUESTION_PATH,
            json={"question": QUESTION},
        )

        assert response.status_code == 200

        assert response.json()["method"] == "hybrid_rerank"

        factory.assert_awaited_once_with()

        pipeline.answer_question.assert_awaited_once_with(
            question=QUESTION,
            method=RetrievalMethod.HYBRID_RERANK,
        )

    finally:
        client.close()
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        (ReadinessState.NOT_STARTED, 503),
        (ReadinessState.PREPARING, 503),
        (ReadinessState.FAILED, 503),
        (ReadinessState.READY, 200),
        (None, 200),
    ],
)
def test_production_factory_is_deferred(
    state: ReadinessState | None,
    expected_status: int,
    tmp_path: Path,
) -> None:
    app = build_app(state, tmp_path)

    pipeline = build_pipeline()

    client = TestClient(app)

    try:
        with patch(
            "backend.app.api.routes.questions.get_question_pipeline",
            return_value=pipeline,
        ) as factory:
            invalid_response = client.post(
                QUESTION_PATH,
                json={"question": "a"},
            )

            assert invalid_response.status_code == 422
            factory.assert_not_called()

            response = client.post(
                QUESTION_PATH,
                json={"question": QUESTION},
            )

            assert response.status_code == expected_status

            if expected_status == 200:
                factory.assert_called_once_with()
                pipeline.answer_question.assert_awaited_once()
            else:
                factory.assert_not_called()
                pipeline.answer_question.assert_not_awaited()

    finally:
        client.close()


def test_openapi_question_request_body_is_preserved(
    tmp_path: Path,
) -> None:
    app = build_app(ReadinessState.READY, tmp_path)

    operation = app.openapi()["paths"][QUESTION_PATH]["post"]

    request_body = operation["requestBody"]

    assert request_body["required"] is True

    assert request_body["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AskRequest",
    }

    assert operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {
        "$ref": "#/components/schemas/AskResponse",
    }


@pytest.mark.asyncio
async def test_question_gate_tracks_live_warmup_transition(
    tmp_path: Path,
) -> None:
    llm_started = asyncio.Event()
    release_llm = asyncio.Event()

    async def warm_up_llm() -> None:
        llm_started.set()
        await release_llm.wait()

    manager = ModelReadinessManager(
        embedding_warmup=Mock(),
        reranker_warmup=Mock(),
        llm_warmup=warm_up_llm,
    )

    settings = Settings(
        _env_file=None,
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        model_warmup_enabled=True,
    )

    app = create_app(
        settings,
        readiness_manager=manager,
    )

    pipeline = build_pipeline()
    factory = Mock(return_value=pipeline)

    app.dependency_overrides[get_question_pipeline] = factory

    try:
        with patch.object(
            app.state.database,
            "initialize",
            new_callable=AsyncMock,
        ):
            async with app.router.lifespan_context(app):
                try:
                    await asyncio.wait_for(
                        llm_started.wait(),
                        timeout=5,
                    )

                    assert manager.snapshot().state == (
                        ReadinessState.PREPARING
                    )

                    transport = httpx.ASGITransport(app=app)

                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        preparing = await asyncio.wait_for(
                            client.post(
                                QUESTION_PATH,
                                json={"question": QUESTION},
                            ),
                            timeout=5,
                        )

                        assert preparing.status_code == 503

                        assert preparing.json()["error"] == (
                            "model_preparing"
                        )

                        factory.assert_not_called()

                        release_llm.set()

                        task = app.state.model_warmup_task

                        assert task is not None

                        await asyncio.wait_for(task, timeout=5)

                        assert manager.snapshot().state == (
                            ReadinessState.READY
                        )

                        ready = await asyncio.wait_for(
                            client.post(
                                QUESTION_PATH,
                                json={"question": QUESTION},
                            ),
                            timeout=5,
                        )

                        assert ready.status_code == 200

                        factory.assert_called_once_with()
                        pipeline.answer_question.assert_awaited_once()

                finally:
                    release_llm.set()

        assert app.state.model_warmup_task is None

    finally:
        release_llm.set()
        app.dependency_overrides.clear()
