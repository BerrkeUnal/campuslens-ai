from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.model_readiness import ReadinessState
from backend.app.rag.retriever import Retriever
from backend.app.schemas.questions import MetricsResponse, PipelineResult
from backend.app.services.retrieval_evaluation import RetrievalEvaluationRunner
from backend.app.services.sqlite_evaluation_service import get_evaluation_runner


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(database_url=f"sqlite:///{(tmp_path / 'evaluation.db').as_posix()}")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def readiness_stub(state: ReadinessState) -> SimpleNamespace:
    return SimpleNamespace(
        snapshot=lambda: SimpleNamespace(state=state),
    )


def test_empty_database_returns_no_evaluation_examples(client: TestClient) -> None:
    response = client.get("/api/evaluations/results")
    assert response.status_code == 200
    assert response.json() == {"results": [], "total": 0}


def test_evaluation_endpoint_persists_measured_results(settings: Settings) -> None:
    app = create_app(settings)
    app.state.model_readiness = readiness_stub(ReadinessState.READY)
    store = create_autospec(VectorStore, instance=True)
    store.search_dense.return_value = []
    assess = AsyncMock(
        return_value=PipelineResult(
            answer="Belgelerde yeterli bilgi yok.",
            insufficient_context=True,
            sources=[],
            metrics=MetricsResponse(
                retrieval_latency_ms=0, generation_latency_ms=0, total_latency_ms=0
            ),
        )
    )
    app.dependency_overrides[get_evaluation_runner] = lambda: (
        lambda: RetrievalEvaluationRunner(
            Retriever(store), is_mock=False, answer_with_context=assess
        )
    )
    with TestClient(app) as client:
        response = client.post("/api/evaluations/run", json={"method": "dense"})
        assert response.status_code == 202
        run = response.json()
        assert run["status"] == "running"
        assert run["split"] == "test"

    assert store.search_dense.await_count == 50
    assert assess.await_count == 8
    with TestClient(create_app(settings)) as client:
        results = client.get("/api/evaluations/results").json()

    assert results["total"] == 1
    (result,) = results["results"]
    assert result["run_id"] == run["run_id"]
    assert result["status"] == "completed" and not result["is_mock"]
    assert result["metrics"]["total_cases"] == 50
    assert result["metrics"]["recall_at_5"] == result["metrics"]["mrr_at_5"] == 0
    assert result["metrics"]["abstention_accuracy"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"method": "unknown_method"},
        {"method": "dense", "split": "development"},
    ],
)
def test_rejects_unknown_evaluation_method_or_split(
    client: TestClient, payload: dict
) -> None:
    response = client.post("/api/evaluations/run", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("readiness_state", "expected_error"),
    [
        (ReadinessState.NOT_STARTED, "model_preparing"),
        (ReadinessState.PREPARING, "model_preparing"),
        (ReadinessState.FAILED, "model_unavailable"),
    ],
)
def test_evaluation_rejects_when_models_are_not_ready(
    settings: Settings,
    readiness_state: ReadinessState,
    expected_error: str,
) -> None:
    app = create_app(settings)
    app.state.model_readiness = readiness_stub(readiness_state)

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/run",
            json={"method": "dense"},
        )
        results = client.get("/api/evaluations/results")

    assert response.status_code == 503
    assert response.json()["error"] == expected_error
    assert results.status_code == 200
    assert results.json() == {"results": [], "total": 0}
