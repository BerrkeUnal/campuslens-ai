from time import perf_counter

from backend.app.rag.hybrid_rerank import HybridRerankSearch
from backend.app.rag.llm_client import LLMGenerator, answer_with_sources
from backend.app.rag.llm_provider import create_llm_client
from backend.app.rag.retriever import Retriever
from backend.app.schemas.questions import MetricsResponse, PipelineResult
from backend.app.schemas.retrieval import (
    RetrievalMethod,
    RetrievedChunk,
    SourceResponse,
)


class RAGQuestionPipeline:
    """Run retrieval, optional reranking and grounded LLM generation."""

    def __init__(
        self,
        retriever: Retriever,
        hybrid_rerank_search: HybridRerankSearch,
        llm_client: LLMGenerator | None = None,
    ) -> None:
        self._retriever = retriever
        self._hybrid_rerank_search = hybrid_rerank_search
        self._llm_client = llm_client

    async def answer_question(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> PipelineResult:
        total_started = perf_counter()
        retrieval_started = perf_counter()

        chunks = await self._retrieve(question, method)
        chunks = chunks[:5]

        retrieval_latency_ms = round((perf_counter() - retrieval_started) * 1000)

        generation_started = perf_counter()

        llm_client = self._get_llm_client() if chunks else None

        output = await answer_with_sources(
            question=question,
            chunks=chunks,
            llm_client=llm_client,
        )

        generation_latency_ms = round((perf_counter() - generation_started) * 1000)
        total_latency_ms = round((perf_counter() - total_started) * 1000)

        sources = self._build_sources(
            chunks=chunks,
            used_source_ids=output.used_source_ids,
        )

        return PipelineResult(
            answer=output.answer,
            insufficient_context=output.insufficient_context,
            sources=sources,
            metrics=MetricsResponse(
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                estimated_cost_usd=None,
            ),
        )

    async def _retrieve(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> list[RetrievedChunk]:
        if method == RetrievalMethod.HYBRID_RERANK:
            return await self._hybrid_rerank_search.search(question)

        return await self._retriever.retrieve(question, method)

    def _get_llm_client(self) -> LLMGenerator:
        if self._llm_client is None:
            self._llm_client = create_llm_client()

        return self._llm_client

    @staticmethod
    def _build_sources(
        chunks: list[RetrievedChunk],
        used_source_ids: list[str],
    ) -> list[SourceResponse]:
        selected_source_ids = set(used_source_ids)
        sources: list[SourceResponse] = []

        for index, chunk in enumerate(chunks[:5], start=1):
            source_id = f"S{index}"

            # Başarılı yanıtta yalnızca LLM'in kullandığı kaynakları göster.
            # Yetersiz bağlamda used_source_ids boş olacağı için incelenen
            # kaynak parçalarının tamamı gösterilir.
            if selected_source_ids and source_id not in selected_source_ids:
                continue

            sources.append(
                SourceResponse(
                    source_id=source_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    source_url=chunk.source_url,
                    page_number=chunk.page_number,
                    excerpt=chunk.text,
                    retrieval_score=chunk.retrieval_score,
                    rerank_score=chunk.rerank_score,
                )
            )

        return sources
