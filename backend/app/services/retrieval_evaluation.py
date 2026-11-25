import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from backend.app.core.exceptions import ApplicationError, ServiceUnavailableError
from backend.app.rag.retriever import Retriever
from backend.app.schemas.questions import PipelineResult
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk
from backend.app.services.evaluation_cases import ResolvedEvaluationCase
from backend.app.services.evaluation_metrics import (
    RetrievalCaseResult,
    RetrievalMetrics,
    source_rank_at_5,
    summarize_retrieval,
)

logger = logging.getLogger(__name__)

Search = Callable[[str, int], Awaitable[list[RetrievedChunk]]]
AnswerWithContext = Callable[[str, list[RetrievedChunk]], Awaitable[PipelineResult]]


@dataclass(frozen=True)
class RetrievalEvaluation:
    cases: list[RetrievalCaseResult]
    metrics: RetrievalMetrics


class RetrievalEvaluationRunner:
    def __init__(
        self,
        retriever: Retriever,
        hybrid_rerank: Search | None = None,
        *,
        is_mock: bool,
        answer_with_context: AnswerWithContext | None = None,
    ) -> None:
        self.is_mock = is_mock
        self._answer_with_context = answer_with_context
        self._retriever = retriever
        self._hybrid_rerank = hybrid_rerank

    async def evaluate(
        self, method: RetrievalMethod, cases: Sequence[ResolvedEvaluationCase]
    ) -> RetrievalEvaluation:
        results = []
        for item in cases:
            start = perf_counter()
            if method == RetrievalMethod.HYBRID_RERANK:
                if self._hybrid_rerank is None:
                    raise ServiceUnavailableError("Hybrid + Rerank")
                sources = await self._hybrid_rerank(item.case.question, 5)
            else:
                sources = await self._retriever.retrieve(item.case.question, method)
            latency = (perf_counter() - start) * 1000
            if item.source_url is not None:
                rank = source_rank_at_5(
                    sources, gold_source_url=item.source_url, gold_pages=item.case.gold_pages
                )
                result = RetrievalCaseResult(item.case.case_id, rank, latency)
            else:
                if self._answer_with_context is None:
                    raise ServiceUnavailableError("Cevapsız soru değerlendirmesi")
                try:
                    response = await self._answer_with_context(item.case.question, sources[:5])
                    result = RetrievalCaseResult(
                        item.case.case_id, None, latency,
                        response.insufficient_context, response.answer,
                    )
                except ApplicationError as exc:
                    logger.warning("Answer evaluation failed for %s: %s", item.case.case_id, exc)
                    result = RetrievalCaseResult(
                        item.case.case_id, None, latency, answer_error=exc.message
                    )
            results.append(result)
        return RetrievalEvaluation(results, summarize_retrieval(results))
