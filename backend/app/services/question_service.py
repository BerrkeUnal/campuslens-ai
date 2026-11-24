from functools import lru_cache
from uuid import uuid4

from backend.app.ports.question_pipeline import QuestionPipeline
from backend.app.rag.pipeline import get_rag_pipeline
from backend.app.schemas.questions import AskRequest, AskResponse


@lru_cache
def get_question_pipeline() -> QuestionPipeline:
    """Build and reuse the configured RAG pipeline."""
    return get_rag_pipeline()


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
