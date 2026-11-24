from functools import lru_cache
from uuid import uuid4

from backend.app.ports.question_pipeline import QuestionPipeline
from backend.app.rag.hybrid_rerank import HybridRerankSearch
from backend.app.rag.qdrant_store import get_vector_store
from backend.app.rag.reranker import get_reranker
from backend.app.rag.retriever import Retriever
from backend.app.schemas.questions import AskRequest, AskResponse
from backend.app.services.rag_question_pipeline import RAGQuestionPipeline


@lru_cache
def get_question_pipeline() -> QuestionPipeline:
    """Build and reuse the real question-answering pipeline."""
    retriever = Retriever(get_vector_store())

    hybrid_rerank_search = HybridRerankSearch(
        retriever=retriever,
        reranker=get_reranker(),
    )

    return RAGQuestionPipeline(
        retriever=retriever,
        hybrid_rerank_search=hybrid_rerank_search,
    )


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
