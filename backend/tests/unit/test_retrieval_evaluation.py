from unittest.mock import AsyncMock, create_autospec

import pytest

from backend.app.core.exceptions import ServiceUnavailableError
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.retriever import Retriever
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk
from backend.app.services.evaluation_cases import load_evaluation_cases
from backend.app.services.retrieval_evaluation import RetrievalEvaluationRunner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "search_name"),
    [
        (RetrievalMethod.BM25, "search_bm25"),
        (RetrievalMethod.DENSE, "search_dense"),
        (RetrievalMethod.HYBRID, "search_hybrid_rrf"),
        (RetrievalMethod.HYBRID_RERANK, "rerank"),
    ],
)
async def test_evaluation_dispatches_search_with_catalog_url(
    method: RetrievalMethod, search_name: str,
) -> None:
    item = load_evaluation_cases()[0]
    store = create_autospec(VectorStore, instance=True)
    rerank = AsyncMock()
    search = rerank if search_name == "rerank" else getattr(store, search_name)
    search.return_value = [
        RetrievedChunk(
            chunk_id="runtime-uuid-p9",
            document_id="runtime-uuid",
            document_title="Yönetmelik",
            source_url=item.source_url,
            page_number=9,
            text="Bir kaynak.",
            retrieval_method=method,
            retrieval_score=1.0,
        ),
    ]
    result = await RetrievalEvaluationRunner(Retriever(store), rerank, is_mock=True).evaluate(method, [item])
    search.assert_awaited_once_with(item.case.question, 5 if search_name == "rerank" else 20)
    assert result.cases[0].case_id == item.case.case_id
    assert result.cases[0].source_rank == 1
    assert result.metrics.recall_at_5 == result.metrics.mrr_at_5 == 1


@pytest.mark.asyncio
async def test_unconfigured_reranker_does_not_create_results() -> None:
    store = create_autospec(VectorStore, instance=True)
    with pytest.raises(ServiceUnavailableError, match="Hybrid"):
        await RetrievalEvaluationRunner(Retriever(store), is_mock=True).evaluate(
            RetrievalMethod.HYBRID_RERANK, load_evaluation_cases()
        )
    store.search_hybrid_rrf.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_answer_generation_keeps_retrieval_metrics() -> None:
    cases = load_evaluation_cases()
    answerable, unanswerable = cases[0], cases[-1]
    store = create_autospec(VectorStore, instance=True)
    store.search_dense.return_value = []
    answer = AsyncMock(side_effect=ServiceUnavailableError("Ollama"))
    runner = RetrievalEvaluationRunner(
        Retriever(store), is_mock=True, answer_with_context=answer
    )

    result = await runner.evaluate(RetrievalMethod.DENSE, [answerable, unanswerable])

    assert result.metrics.total_cases == 2
    assert result.metrics.recall_at_5 == 0
    assert result.metrics.abstention_accuracy is None
    assert result.cases[-1].insufficient_context is None
    assert "Ollama" in (result.cases[-1].answer_error or "")
    assert not result.cases[-1].answerable
