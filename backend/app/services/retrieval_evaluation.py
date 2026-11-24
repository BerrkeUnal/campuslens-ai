from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from backend.app.core.exceptions import ServiceUnavailableError
from backend.app.ports.vector_store import VectorStore
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk
from backend.app.services.evaluation_cases import ResolvedEvaluationCase
from backend.app.services.evaluation_metrics import (
    RetrievalCaseResult,
    RetrievalMetrics,
    source_rank_at_5,
    summarize_retrieval,
)

Search = Callable[[str, int], Awaitable[list[RetrievedChunk]]]


@dataclass(frozen=True)
class RetrievalEvaluation:
    cases: list[RetrievalCaseResult]
    metrics: RetrievalMetrics


class RetrievalEvaluationRunner:
    def __init__(
        self, store: VectorStore, hybrid_rerank: Search | None = None, *, is_mock: bool
    ) -> None:
        self.is_mock = is_mock
        self._searches = {
            RetrievalMethod.BM25: store.search_bm25,
            RetrievalMethod.DENSE: store.search_dense,
            RetrievalMethod.HYBRID: store.search_hybrid_rrf,
            RetrievalMethod.HYBRID_RERANK: hybrid_rerank,
        }

    async def evaluate(
        self, method: RetrievalMethod, cases: Sequence[ResolvedEvaluationCase]
    ) -> RetrievalEvaluation:
        search = self._searches[method]
        if search is None:
            raise ServiceUnavailableError("Hybrid + Rerank")
        results = []
        for item in cases:
            start = perf_counter()
            sources = await search(item.case.question, 5)
            latency = (perf_counter() - start) * 1000
            rank = source_rank_at_5(
                sources, gold_source_url=item.source_url, gold_pages=item.case.gold_pages
            )
            results.append(RetrievalCaseResult(item.case.case_id, rank, latency))
        return RetrievalEvaluation(results, summarize_retrieval(results))
