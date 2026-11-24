from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.questions import MetricsResponse, PipelineResult
from backend.app.schemas.retrieval import RetrievalMethod, SourceResponse
from backend.app.services.question_service import get_question_pipeline


class FakeQuestionPipeline:
    async def answer_question(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> PipelineResult:
        return PipelineResult(
            answer=f'"{question}" sorusunun test yanıtıdır. [S1]',
            insufficient_context=False,
            sources=[
                SourceResponse(
                    source_id="S1",
                    chunk_id="test-document-p4-c1",
                    document_id="test-document",
                    document_title="Önlisans ve Lisans Yönetmeliği",
                    source_url="https://www.example.edu.tr/regulation.pdf",
                    page_number=4,
                    excerpt="Testlerde kullanılan kaynak metnidir.",
                    retrieval_score=0.95,
                    rerank_score=(
                        0.98 if method == RetrievalMethod.HYBRID_RERANK else None
                    ),
                )
            ],
            metrics=MetricsResponse(
                retrieval_latency_ms=10,
                generation_latency_ms=20,
                total_latency_ms=30,
                estimated_cost_usd=None,
            ),
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    pipeline = FakeQuestionPipeline()

    def override_question_pipeline() -> FakeQuestionPipeline:
        return pipeline

    app.dependency_overrides[get_question_pipeline] = override_question_pipeline

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_ask_question_returns_grounded_response(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/questions/ask",
        json={
            "question": "Bütünleme sınavına kimler girebilir?",
            "method": "hybrid_rerank",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["method"] == "hybrid_rerank"
    assert body["insufficient_context"] is False
    assert "[S1]" in body["answer"]
    assert body["sources"][0]["source_id"] == "S1"
    assert body["sources"][0]["rerank_score"] == 0.98
    assert body["metrics"]["total_latency_ms"] == 30
    assert body["request_id"]


def test_ask_question_rejects_short_question(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/questions/ask",
        json={
            "question": "a",
            "method": "hybrid_rerank",
        },
    )

    assert response.status_code == 422


def test_ask_question_rejects_unknown_method(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/questions/ask",
        json={
            "question": "Bütünleme sınavı ne zaman?",
            "method": "unknown_method",
        },
    )

    assert response.status_code == 422
