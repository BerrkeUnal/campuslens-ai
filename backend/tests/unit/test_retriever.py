from unittest.mock import AsyncMock

import pytest

from backend.app.core.config import get_settings
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.retriever import Retriever
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk


@pytest.fixture
def vector_store():
    return AsyncMock(spec=VectorStore)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "search_function", "limit_setting"),
    [
        (RetrievalMethod.BM25, "search_bm25", "bm25_top_k"),
        (RetrievalMethod.DENSE, "search_dense", "dense_top_k"),
        (RetrievalMethod.HYBRID, "search_hybrid_rrf", "dense_top_k"),
        (RetrievalMethod.HYBRID_RERANK, "search_hybrid_rrf", "dense_top_k"),
    ],
)
async def test_retrieval_method_routing(
    vector_store,
    method,
    search_function,
    limit_setting,
):
    question = "Bütünleme sınavına kimler girebilir?"
    settings = get_settings()

    candidate = RetrievedChunk(
        chunk_id="doc1-p1-c1",
        document_id="doc1",
        document_title="Sınav Yönetmeliği",
        source_url="https://example.com/regulation",
        page_number=1,
        text="Bütünleme sınavına ilişkin kurallar.",
        retrieval_method=(
            RetrievalMethod.HYBRID
            if method == RetrievalMethod.HYBRID_RERANK
            else method
        ),
        retrieval_score=0.85,
    )

    selected_search = getattr(vector_store, search_function)
    selected_search.return_value = [candidate]

    retriever = Retriever(vector_store)
    result = await retriever.retrieve(question, method)

    selected_search.assert_awaited_once_with(
        question,
        getattr(settings, limit_setting),
    )

    for name in (
        "search_bm25",
        "search_dense",
        "search_hybrid_rrf",
    ):
        if name != search_function:
            getattr(vector_store, name).assert_not_awaited()

    assert result == [candidate]
    assert result[0].retrieval_score == 0.85
    assert result[0].rerank_score is None


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["", "   "])
async def test_empty_question_is_rejected(vector_store, question):
    retriever = Retriever(vector_store)

    with pytest.raises(ValueError, match="Question must not be empty"):
        await retriever.retrieve(question, RetrievalMethod.BM25)

    vector_store.search_bm25.assert_not_awaited()
    vector_store.search_dense.assert_not_awaited()
    vector_store.search_hybrid_rrf.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsupported_method_is_rejected(vector_store):
    retriever = Retriever(vector_store)

    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        await retriever.retrieve("Test question", "invalid")

    vector_store.search_bm25.assert_not_awaited()
    vector_store.search_dense.assert_not_awaited()
    vector_store.search_hybrid_rrf.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_retrieval_result(vector_store):
    vector_store.search_dense.return_value = []
    retriever = Retriever(vector_store)

    result = await retriever.retrieve(
        "Test question",
        RetrievalMethod.DENSE,
    )

    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "search_function"),
    [
        (RetrievalMethod.BM25, "search_bm25"),
        (RetrievalMethod.DENSE, "search_dense"),
        (RetrievalMethod.HYBRID, "search_hybrid_rrf"),
        (RetrievalMethod.HYBRID_RERANK, "search_hybrid_rrf"),
    ],
)
async def test_retrieval_never_exceeds_twenty_candidates(
    vector_store,
    monkeypatch,
    method,
    search_function,
):
    settings = get_settings()

    monkeypatch.setattr(settings, "bm25_top_k", 50)
    monkeypatch.setattr(settings, "dense_top_k", 50)

    candidates = [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document_id="doc-1",
            document_title="Test document",
            source_url="https://example.com/regulation",
            page_number=1,
            text=f"Test source {i}",
            retrieval_method=method,
            retrieval_score=1.0 / i,
        )
        for i in range(1, 51)
    ]

    selected_search = getattr(vector_store, search_function)
    selected_search.return_value = candidates

    retriever = Retriever(vector_store)

    result = await retriever.retrieve(
        "Test question",
        method,
    )

    selected_search.assert_awaited_once_with(
        "Test question",
        20,
    )

    assert len(result) == 20
    assert result == candidates[:20]
