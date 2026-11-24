import asyncio

from backend.app.core.config import get_settings
from backend.app.rag.qdrant_store import get_vector_store
from backend.app.rag.reranker import Reranker, get_reranker
from backend.app.rag.retriever import Retriever
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk


class HybridRerankSearch:
    """Combine hybrid retrieval with cross-encoder reranking."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker

    async def search(
        self,
        question: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve hybrid candidates and return reranked results."""
        if not question.strip():
            raise ValueError("Question must not be empty.")

        if limit < 1:
            raise ValueError("limit must be positive.")

        candidates = await self._retriever.retrieve(
            question,
            RetrievalMethod.HYBRID_RERANK,
        )

        if not candidates:
            return []

        return await asyncio.to_thread(
            self._reranker.rerank,
            question,
            candidates,
            limit,
        )


async def search_hybrid_rerank(
    question: str,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Provide a reranked search adapter for pipeline and evaluation."""
    if limit is None:
        limit = get_settings().rerank_top_k

    search = HybridRerankSearch(
        retriever=Retriever(get_vector_store()),
        reranker=get_reranker(),
    )

    return await search.search(question, limit)
