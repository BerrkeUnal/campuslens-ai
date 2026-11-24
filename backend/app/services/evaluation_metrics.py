"""Metrics for ranking evidence pages in search results."""

from collections.abc import Collection, Sequence

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
