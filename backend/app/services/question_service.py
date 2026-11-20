from uuid import uuid4

from backend.app.ports.question_pipeline import QuestionPipeline
from backend.app.schemas.questions import (
    AskRequest,
    AskResponse,
    MetricsResponse,
    PipelineResult,
)
from backend.app.schemas.retrieval import (
    RetrievalMethod,
    SourceResponse,
)


class MockQuestionPipeline:
    async def answer_question(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> PipelineResult:
        rerank_score = 1.0 if method == RetrievalMethod.HYBRID_RERANK else None

        source = SourceResponse(
            source_id="S1",
            chunk_id="mock-document-p4-c1",
            document_id="mock-document",
            document_title="Önlisans ve Lisans Yönetmeliği",
            source_url="https://www.example.edu.tr/regulations",
            page_number=4,
            excerpt=("Bu yanıt geliştirme amaçlı mock kaynaktan oluşturulmuştur."),
            retrieval_score=1.0,
            rerank_score=rerank_score,
        )

        return PipelineResult(
            answer=(
                f'"{question}" sorusu için oluşturulmuş geçici backend yanıtıdır. [S1]'
            ),
            insufficient_context=False,
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
    result = await pipeline.answer_question(
        question=request.question,
        method=request.method,
    )

    return AskResponse(
        request_id=uuid4(),
        answer=result.answer,
        insufficient_context=result.insufficient_context,
        method=request.method,
        sources=result.sources,
        metrics=result.metrics,
    )
