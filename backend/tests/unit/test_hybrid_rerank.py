from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.app.core.config import get_settings
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.hybrid_rerank import (
    HybridRerankSearch,
    search_hybrid_rerank,
)
from backend.app.rag.reranker import Reranker
from backend.app.rag.retriever import Retriever
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk


@pytest.fixture
def candidates():
    return [
        RetrievedChunk(
            chunk_id=f"doc1-p{i}-c1",
            document_id="doc1",
            document_title="Sınav Yönetmeliği",
            source_url="https://example.com/regulation",
            page_number=i,
            text=f"Kaynak metni {i}",
            retrieval_method=RetrievalMethod.HYBRID,
            retrieval_score=1.0 / i,
        )
        for i in range(1, 21)
    ]


@pytest.fixture
def vector_store():
    return AsyncMock(spec=VectorStore)


@pytest.fixture
def reranker():
    return Mock(spec=Reranker)


@pytest.mark.asyncio
async def test_hybrid_retrieval_and_reranking(candidates, vector_store, reranker):
    question = "Bütünleme sınavına kimler girebilir?"
    vector_store.search_hybrid_rrf.return_value = candidates

    expected = [
        item.model_copy(update={"rerank_score": 0.9}) for item in candidates[-5:]
    ]
    reranker.rerank.return_value = expected

    search = HybridRerankSearch(Retriever(vector_store), reranker)
    result = await search.search(question, 5)

    vector_store.search_hybrid_rrf.assert_awaited_once_with(
        question, get_settings().dense_top_k
    )
    vector_store.search_bm25.assert_not_awaited()
    vector_store.search_dense.assert_not_awaited()

    reranker.rerank.assert_called_once_with(question, candidates, 5)

    assert result == expected
    assert len(result) == 5


@pytest.mark.asyncio
async def test_empty_candidates_skip_reranker(vector_store, reranker):
    vector_store.search_hybrid_rrf.return_value = []

    search = HybridRerankSearch(Retriever(vector_store), reranker)

    result = await search.search("Test question", 5)

    assert result == []
    reranker.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_custom_result_limit(candidates, vector_store, reranker):
    vector_store.search_hybrid_rrf.return_value = candidates
    reranker.rerank.return_value = candidates[:3]

    search = HybridRerankSearch(Retriever(vector_store), reranker)

    result = await search.search("Test question", 3)

    reranker.rerank.assert_called_once_with("Test question", candidates, 3)
    assert len(result) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["", "   "])
async def test_empty_question_is_rejected(question, vector_store, reranker):
    search = HybridRerankSearch(Retriever(vector_store), reranker)

    with pytest.raises(ValueError, match="Question must not be empty"):
        await search.search(question, 5)

    vector_store.search_hybrid_rrf.assert_not_awaited()
    reranker.rerank.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_invalid_limit_is_rejected(limit, vector_store, reranker):
    search = HybridRerankSearch(Retriever(vector_store), reranker)

    with pytest.raises(ValueError, match="limit must be positive"):
        await search.search("Test question", limit)

    vector_store.search_hybrid_rrf.assert_not_awaited()
    reranker.rerank.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 3])
async def test_public_search_adapter(limit, candidates, vector_store, reranker):
    expected_limit = get_settings().rerank_top_k if limit is None else limit

    vector_store.search_hybrid_rrf.return_value = candidates
    reranker.rerank.return_value = candidates[:expected_limit]

    with (
        patch(
            "backend.app.rag.hybrid_rerank.get_vector_store",
            return_value=vector_store,
        ),
        patch(
            "backend.app.rag.hybrid_rerank.get_reranker",
            return_value=reranker,
        ),
    ):
        result = await search_hybrid_rerank("Test question", limit)

    reranker.rerank.assert_called_once_with("Test question", candidates, expected_limit)

    assert result == candidates[:expected_limit]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        RetrievalMethod.BM25,
        RetrievalMethod.DENSE,
        RetrievalMethod.HYBRID,
    ],
)
async def test_standard_retrieval_does_not_load_reranker(
    method, candidates, vector_store
):
    vector_store.search_bm25.return_value = candidates
    vector_store.search_dense.return_value = candidates
    vector_store.search_hybrid_rrf.return_value = candidates

    with patch("backend.app.rag.reranker.CrossEncoder") as cross_encoder:
        result = await Retriever(vector_store).retrieve("Test question", method)

    assert result == candidates
    cross_encoder.assert_not_called()
