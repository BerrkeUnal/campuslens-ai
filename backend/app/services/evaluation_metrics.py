"""Metrics for ranking evidence pages in search results."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from backend.app.schemas.retrieval import RetrievedChunk, SourceResponse


def source_rank_at_5(
    sources: Sequence[RetrievedChunk | SourceResponse],
    *,
    gold_source_url: str,
    gold_pages: Collection[int],
) -> int | None:
    """Return the first matching PDF page's rank in the top five, if present."""
    for rank, source in enumerate(sources[:5], start=1):
        if source.source_url == gold_source_url and source.page_number in gold_pages:
            return rank
    return None


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    source_rank: int | None
    latency_ms: float
    insufficient_context: bool | None = None
    answer: str | None = None
    answer_error: str | None = None

    @property
    def answerable(self) -> bool:
        return self.insufficient_context is None and self.answer_error is None

    @property
    def source_hit(self) -> bool:
        return self.source_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.source_rank if self.source_rank is not None else 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    total_cases: int
    recall_at_5: float
    mrr_at_5: float
    average_latency_ms: float
    abstention_accuracy: float | None = None


def summarize_retrieval(results: Sequence[RetrievalCaseResult]) -> RetrievalMetrics:
    total = len(results)
    if total == 0:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0)
    answerable = [result for result in results if result.answerable]
    abstentions = [result.insufficient_context for result in results if result.insufficient_context is not None]
    return RetrievalMetrics(
        total_cases=total,
        recall_at_5=sum(result.source_hit for result in answerable) / len(answerable) if answerable else 0.0,
        mrr_at_5=sum(result.reciprocal_rank for result in answerable) / len(answerable) if answerable else 0.0,
        average_latency_ms=sum(result.latency_ms for result in results) / total,
        abstention_accuracy=sum(abstentions) / len(abstentions) if abstentions else None,
    )
