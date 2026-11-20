from typing import Protocol
from uuid import uuid4

from backend.app.schemas.questions import (
    AskRequest,
    AskResponse,
    MetricsResponse,
)
from backend.app.schemas.retrieval import (
    RetrievalMethod,
    SourceResponse,
)


class QuestionPipeline(Protocol):
    async def answer_question(
        self,
        request: AskRequest,
    ) -> AskResponse: ...


class MockQuestionPipeline:
    async def answer_question(
        self,
        request: AskRequest,
    ) -> AskResponse:
        rerank_score = 1.0 if request.method == RetrievalMethod.HYBRID_RERANK else None

        source = SourceResponse(
            source_id="S1",
            chunk_id="mock-document-p4-c1",
            document_id="mock-document",
            document_title="Önlisans ve Lisans Yönetmeliği",
            source_url="https://www.example.edu.tr/regulations",
            page_number=4,
            excerpt="Bu yanıt geliştirme amaçlı mock kaynaktan oluşturulmuştur.",
            retrieval_score=1.0,
            rerank_score=rerank_score,
        )

        return AskResponse(
            request_id=uuid4(),
            answer=(
                "Bu, backend entegrasyonu için oluşturulmuş geçici bir yanıttır. [S1]"
            ),
            insufficient_context=False,
            method=request.method,
            sources=[source],
            metrics=MetricsResponse(
                retrieval_latency_ms=0,
                generation_latency_ms=0,
                total_latency_ms=0,
                estimated_cost_usd=0,
            ),
        )


def get_question_pipeline() -> QuestionPipeline:
    return MockQuestionPipeline()


async def ask_question(
    request: AskRequest,
    pipeline: QuestionPipeline,
) -> AskResponse:
    return await pipeline.answer_question(request)
