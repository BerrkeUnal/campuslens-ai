from fastapi.testclient import TestClient

from backend.app.core.exceptions import ServiceUnavailableError
from backend.app.main import app
from backend.app.schemas.questions import PipelineResult
from backend.app.schemas.retrieval import RetrievalMethod
from backend.app.services.question_service import (
    get_question_pipeline,
)

client = TestClient(app)


EXPECTED_PATHS = {
    "/api/health",
    "/api/documents/upload",
    "/api/documents",
    "/api/documents/{document_id}",
    "/api/questions/ask",
    "/api/evaluations/run",
    "/api/evaluations/results",
}


class FailingQuestionPipeline:
    async def answer_question(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> PipelineResult:
        raise ServiceUnavailableError("LLM")


def get_failing_question_pipeline() -> FailingQuestionPipeline:
    return FailingQuestionPipeline()


def test_openapi_contains_all_required_paths() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = set(response.json()["paths"])

    assert EXPECTED_PATHS.issubset(paths)


def test_response_contains_request_id_header() -> None:
    response = client.get(
        "/api/health",
        headers={
            "X-Request-ID": "integration-test-request",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "integration-test-request"


def test_validation_error_uses_common_format() -> None:
    response = client.post(
        "/api/questions/ask",
        json={
            "question": "a",
            "method": "unknown",
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["error"] == "validation_error"
    assert body["message"] == "İstek alanlarından biri geçersiz."
    assert body["request_id"]
    assert isinstance(body["details"], list)


def test_service_error_returns_503() -> None:
    app.dependency_overrides[get_question_pipeline] = get_failing_question_pipeline

    try:
        response = client.post(
            "/api/questions/ask",
            json={
                "question": "Bütünleme sınavı ne zaman?",
                "method": "hybrid_rerank",
            },
        )
    finally:
        app.dependency_overrides.pop(
            get_question_pipeline,
            None,
        )

    assert response.status_code == 503

    body = response.json()

    assert body["error"] == "service_unavailable"
    assert body["request_id"]
    assert "LLM" in body["message"]


def test_llm_client_error_returns_503() -> None:
    from backend.app.rag.llm_client import LLMClientError

    class FailingLLMPipeline:
        async def answer_question(
            self,
            question: str,
            method: RetrievalMethod,
        ) -> PipelineResult:
            raise LLMClientError("LLM servisine bağlanılamadı.")

    app.dependency_overrides[get_question_pipeline] = FailingLLMPipeline

    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post(
                "/api/questions/ask",
                json={
                    "question": "Bütünleme sınavı ne zaman?",
                    "method": "hybrid_rerank",
                },
            )
    finally:
        app.dependency_overrides.pop(get_question_pipeline, None)

    assert response.status_code == 503

    body = response.json()

    assert body["error"] == "service_unavailable"
    assert body["request_id"]
    assert "LLM" in body["message"]
    assert body["details"] is None
