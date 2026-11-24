from backend.app.core.config import get_settings
from backend.app.ports.vector_store import VectorStore
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk


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
        """Retrieve candidates using the selected search method."""
        if not question.strip():
            raise ValueError("Question must not be empty.")

        if method == RetrievalMethod.BM25:
            return await self._vector_store.search_bm25(
                question,
                self._settings.bm25_top_k,
            )

        if method == RetrievalMethod.DENSE:
            return await self._vector_store.search_dense(
                question,
                self._settings.dense_top_k,
            )

        if method in (
            RetrievalMethod.HYBRID,
            RetrievalMethod.HYBRID_RERANK,
        ):
            return await self._vector_store.search_hybrid_rrf(
                question,
                self._settings.dense_top_k,
            )

        raise ValueError(f"Unsupported retrieval method: {method}")
