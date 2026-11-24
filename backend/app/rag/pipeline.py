from collections.abc import Callable
from time import perf_counter

from backend.app.rag.cost_estimator import estimate_cost
from backend.app.rag.hybrid_rerank import HybridRerankSearch
from backend.app.rag.llm_client import LLMClientError, LLMGenerator
from backend.app.rag.llm_provider import create_llm_client
from backend.app.rag.prompt_builder import build_prompt
from backend.app.rag.qdrant_store import get_vector_store
from backend.app.rag.reranker import get_reranker
from backend.app.rag.retriever import Retriever
from backend.app.rag.source_validator import (
    SourceValidationError,
    validate_sources,
)
from backend.app.schemas.questions import (
    AskRequest,
    MetricsResponse,
    PipelineResult,
)
from backend.app.schemas.retrieval import (
    RetrievalMethod,
    SourceResponse,
)


class RAGPipeline:
    """Coordinate retrieval, generation and source validation."""

    def __init__(
        self,
        retriever: Retriever,
        hybrid_search: HybridRerankSearch,
        llm_factory: Callable[[], LLMGenerator] = create_llm_client,
    ) -> None:
        self._retriever = retriever
        self._hybrid_search = hybrid_search
        self._llm_factory = llm_factory

    async def answer_question(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> PipelineResult:
        """Return a grounded answer with sources and metrics."""

        total_start = perf_counter()

        request = AskRequest(
            question=question,
            method=method,
        )

        question = request.question
        method = request.method

        retrieval_start = perf_counter()

        if method == RetrievalMethod.HYBRID_RERANK:
            candidates = await self._hybrid_search.search(
                question,
                5,
            )
        else:
            candidates = await self._retriever.retrieve(
                question,
                method,
            )

        context = candidates[:5]

        retrieval_latency_ms = int((perf_counter() - retrieval_start) * 1000)

        if not context:
            return PipelineResult(
                answer=("Bu soruyu yanıtlamak için yeterli kaynak bulunamadı."),
                insufficient_context=True,
                sources=[],
                metrics=MetricsResponse(
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=0,
                    total_latency_ms=int((perf_counter() - total_start) * 1000),
                    estimated_cost_usd=0.0,
                    input_tokens=0,
                    output_tokens=0,
                ),
            )

        prompt, source_map = build_prompt(question, context)

        generation_start = perf_counter()

        llm_client = self._llm_factory()

        try:
            generation = await llm_client.generate_with_usage(prompt)
        finally:
            await llm_client.aclose()

        generation_latency_ms = int((perf_counter() - generation_start) * 1000)

        try:
            output = validate_sources(
                generation.output,
                source_map,
            )
        except SourceValidationError as exc:
            raise LLMClientError(
                "LLM yanıtındaki kaynak doğrulaması başarısız oldu."
            ) from exc

        sources = []
        source_ids = output.used_source_ids

        if output.insufficient_context and not source_ids:
            source_ids = list(source_map)

        for source_id in source_ids:
            chunk = source_map[source_id]

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

        usage = generation.usage

        if getattr(llm_client, "provider", "openai") == "ollama":
            estimated_cost = 0.0
        else:
            estimated_cost = estimate_cost(
                model=llm_client.model,
                input_tokens=(usage.input_tokens if usage is not None else None),
                output_tokens=(usage.output_tokens if usage is not None else None),
            )

        return PipelineResult(
            answer=output.answer,
            insufficient_context=output.insufficient_context,
            sources=sources,
            metrics=MetricsResponse(
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=int((perf_counter() - total_start) * 1000),
                estimated_cost_usd=estimated_cost,
                input_tokens=(usage.input_tokens if usage is not None else None),
                output_tokens=(usage.output_tokens if usage is not None else None),
            ),
        )


def get_rag_pipeline() -> RAGPipeline:
    """Build a pipeline using the shared retrieval dependencies."""

    retriever = Retriever(get_vector_store())

    hybrid_search = HybridRerankSearch(
        retriever=retriever,
        reranker=get_reranker(),
    )

    return RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
    )


async def answer_question(
    question: str,
    method: RetrievalMethod,
) -> PipelineResult:
    """Answer a question using the configured RAG pipeline."""

    pipeline = get_rag_pipeline()

    return await pipeline.answer_question(
        question,
        method,
    )
