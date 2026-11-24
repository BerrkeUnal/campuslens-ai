from backend.app.core.config import get_settings
from backend.app.ports.vector_store import VectorStore
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk

MAX_RETRIEVAL_CANDIDATES = 20


class Retriever:
    """Route retrieval requests to the configured vector store."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store
        self._settings = get_settings()

    async def retrieve(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> list[RetrievedChunk]:
        """Retrieve at most twenty candidates using the selected method."""
        if not question.strip():
            raise ValueError("Question must not be empty.")

        if method == RetrievalMethod.BM25:
            limit = min(
                self._settings.bm25_top_k,
                MAX_RETRIEVAL_CANDIDATES,
            )
            candidates = await self._vector_store.search_bm25(
                question,
                limit,
            )

        elif method == RetrievalMethod.DENSE:
            limit = min(
                self._settings.dense_top_k,
                MAX_RETRIEVAL_CANDIDATES,
            )
            candidates = await self._vector_store.search_dense(
                question,
                limit,
            )

        elif method in (
            RetrievalMethod.HYBRID,
            RetrievalMethod.HYBRID_RERANK,
        ):
            limit = min(
                self._settings.dense_top_k,
                MAX_RETRIEVAL_CANDIDATES,
            )
            candidates = await self._vector_store.search_hybrid_rrf(
                question,
                limit,
            )

        else:
            raise ValueError(f"Unsupported retrieval method: {method}")

        return candidates[:limit]
