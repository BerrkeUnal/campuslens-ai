import pytest

from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk
from backend.app.services.evaluation_metrics import (
    RetrievalCaseResult,
    RetrievalMetrics,
    source_rank_at_5,
    summarize_retrieval,
)

GOLD_URL = "https://example.edu.tr/regulation.pdf"
OTHER_URL = "https://example.edu.tr/other.pdf"


def make_chunk(url: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{url}#page={page}",
        document_id="upload-id",
        document_title="Example document",
        source_url=url,
        page_number=page,
        text="Example source passage.",
        retrieval_method=RetrievalMethod.DENSE,
        retrieval_score=0.5,
    )


def test_fifth_result_matches_document_and_one_of_the_gold_pages() -> None:
    sources = [
        make_chunk(OTHER_URL, 9),
        make_chunk(GOLD_URL, 8),
        make_chunk(OTHER_URL, 11),
        make_chunk(GOLD_URL, 7),
        make_chunk(GOLD_URL, 11),
        make_chunk(GOLD_URL, 9),
    ]

    assert source_rank_at_5(sources, gold_source_url=GOLD_URL, gold_pages=[9, 11]) == 5


def test_sixth_result_is_outside_the_cutoff() -> None:
    sources = [make_chunk(GOLD_URL, page) for page in [1, 2, 3, 4, 5, 9]]

    assert source_rank_at_5(sources, gold_source_url=GOLD_URL, gold_pages=[9]) is None


def test_wrong_url_wrong_page_and_empty_results_are_misses() -> None:
    for sources in ([make_chunk(OTHER_URL, 9)], [make_chunk(GOLD_URL, 10)], []):
        assert source_rank_at_5(sources, gold_source_url=GOLD_URL, gold_pages=[9]) is None


def test_recall_and_mrr_include_misses_in_denominator() -> None:
    results = [
        RetrievalCaseResult("first", 1, 10.0),
        RetrievalCaseResult("fifth", 5, 20.0),
        RetrievalCaseResult("miss", None, 30.0),
    ]
    metrics = summarize_retrieval(results)
    assert metrics.total_cases == 3
    assert metrics.recall_at_5 == pytest.approx(2 / 3)
    assert metrics.mrr_at_5 == pytest.approx((1 + 1 / 5) / 3)
    assert metrics.average_latency_ms == 20.0
    assert not results[2].source_hit
    assert results[2].reciprocal_rank == 0.0


def test_empty_case_set_has_zero_metrics() -> None:
    assert summarize_retrieval([]) == RetrievalMetrics(0, 0.0, 0.0, 0.0)
