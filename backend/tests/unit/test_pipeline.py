from unittest.mock import AsyncMock, Mock

import pytest

from backend.app.rag.hybrid_rerank import HybridRerankSearch
from backend.app.rag.llm_client import (
    LLMClient,
    LLMGenerationResult,
    LLMOutput,
    LLMUsage,
)
from backend.app.rag.pipeline import RAGPipeline
from backend.app.rag.retriever import Retriever
from backend.app.schemas.retrieval import (
    RetrievalMethod,
    RetrievedChunk,
)


@pytest.fixture
def candidates():
    return [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document_id="doc-1",
            document_title="Sinav Yonetmeligi",
            source_url="https://example.edu.tr/sinav.pdf",
            page_number=i,
            section_title="Butunleme Sinavi",
            text=f"Test source content {i}",
            retrieval_method=RetrievalMethod.HYBRID,
            retrieval_score=1.0 / i,
        )
        for i in range(1, 21)
    ]


@pytest.fixture
def dependencies(candidates):
    retriever = AsyncMock(spec=Retriever)
    retriever.retrieve.return_value = candidates

    hybrid_search = AsyncMock(spec=HybridRerankSearch)
    hybrid_search.search.return_value = candidates[:5]

    llm_client = AsyncMock(spec=LLMClient)
    llm_client.model = "gpt-4.1-mini"

    llm_client.generate_with_usage.return_value = LLMGenerationResult(
        output=LLMOutput(
            answer="Kaynakli yanit. [S1]",
            used_source_ids=["S1"],
            insufficient_context=False,
        ),
        usage=LLMUsage(
            input_tokens=1200,
            output_tokens=80,
        ),
    )

    llm_factory = Mock(return_value=llm_client)

    return retriever, hybrid_search, llm_client, llm_factory


@pytest.mark.asyncio
@pytest.mark.parametrize("method", list(RetrievalMethod))
async def test_pipeline_routes_retrieval_method(
    method,
    dependencies,
):
    retriever, hybrid_search, llm_client, llm_factory = dependencies

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    result = await pipeline.answer_question(
        "Butunleme sinavina kimler girebilir?",
        method,
    )

    if method == RetrievalMethod.HYBRID_RERANK:
        hybrid_search.search.assert_awaited_once_with(
            "Butunleme sinavina kimler girebilir?",
            5,
        )
        retriever.retrieve.assert_not_awaited()
    else:
        retriever.retrieve.assert_awaited_once_with(
            "Butunleme sinavina kimler girebilir?",
            method,
        )
        hybrid_search.search.assert_not_awaited()

    llm_client.generate_with_usage.assert_awaited_once()

    assert result.insufficient_context is False
    assert result.sources[0].source_id == "S1"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", list(RetrievalMethod))
async def test_empty_context_skips_llm(
    method,
    dependencies,
):
    retriever, hybrid_search, llm_client, llm_factory = dependencies

    retriever.retrieve.return_value = []
    hybrid_search.search.return_value = []

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    result = await pipeline.answer_question(
        "Butunleme sinavina kimler girebilir?",
        method,
    )

    assert result.insufficient_context is True
    assert result.sources == []
    assert result.metrics.generation_latency_ms == 0
    assert result.metrics.estimated_cost_usd == 0.0

    llm_factory.assert_not_called()
    llm_client.generate_with_usage.assert_not_awaited()


def test_pipeline_factory_wires_dependencies(monkeypatch):
    from backend.app.rag import pipeline as pipeline_module

    vector_store = Mock()
    reranker = Mock()
    retriever = Mock()
    hybrid_search = Mock()

    vector_factory = Mock(return_value=vector_store)
    reranker_factory = Mock(return_value=reranker)
    retriever_factory = Mock(return_value=retriever)
    hybrid_factory = Mock(return_value=hybrid_search)

    monkeypatch.setattr(
        pipeline_module,
        "get_vector_store",
        vector_factory,
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_reranker",
        reranker_factory,
    )
    monkeypatch.setattr(
        pipeline_module,
        "Retriever",
        retriever_factory,
    )
    monkeypatch.setattr(
        pipeline_module,
        "HybridRerankSearch",
        hybrid_factory,
    )

    result = pipeline_module.get_rag_pipeline()

    assert isinstance(result, RAGPipeline)

    vector_factory.assert_called_once_with()
    reranker_factory.assert_called_once_with()

    retriever_factory.assert_called_once_with(
        vector_store,
    )

    hybrid_factory.assert_called_once_with(
        retriever=retriever,
        reranker=reranker,
    )


@pytest.mark.asyncio
async def test_public_answer_question(monkeypatch):
    from backend.app.rag import pipeline as pipeline_module

    expected_result = Mock()

    pipeline = Mock()
    pipeline.answer_question = AsyncMock(
        return_value=expected_result,
    )

    factory = Mock(return_value=pipeline)

    monkeypatch.setattr(
        pipeline_module,
        "get_rag_pipeline",
        factory,
    )

    question = "Butunleme sinavina kimler girebilir?"
    method = RetrievalMethod.HYBRID_RERANK

    result = await pipeline_module.answer_question(
        question,
        method,
    )

    factory.assert_called_once_with()

    pipeline.answer_question.assert_awaited_once_with(
        question,
        method,
    )

    assert result is expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "used_source_ids"),
    [
        ("Uydurma kaynak. [S9]", ["S9"]),
        ("Gecersiz citation. [S9]", ["S1"]),
        ("Uyusmayan kaynak. [S1]", ["S2"]),
        ("Kaynak listesi eksik. [S1]", []),
        ("Kaynaksiz kesin yanit.", []),
        ("Context disindaki kaynak. [S6]", ["S6"]),
    ],
)
async def test_pipeline_rejects_invalid_sources(
    answer,
    used_source_ids,
    dependencies,
):
    from backend.app.rag.llm_client import LLMClientError
    from backend.app.rag.source_validator import SourceValidationError

    retriever, hybrid_search, llm_client, llm_factory = dependencies

    llm_client.generate_with_usage.return_value = LLMGenerationResult(
        output=LLMOutput(
            answer=answer,
            used_source_ids=used_source_ids,
            insufficient_context=False,
        ),
        usage=LLMUsage(
            input_tokens=1200,
            output_tokens=80,
        ),
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    with pytest.raises(
        LLMClientError,
        match="kaynak doğrulaması başarısız oldu",
    ) as exc_info:
        await pipeline.answer_question(
            "Butunleme sinavina kimler girebilir?",
            RetrievalMethod.HYBRID,
        )

    assert isinstance(
        exc_info.value.__cause__,
        SourceValidationError,
    )

    llm_client.generate_with_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_maps_actual_source_metadata(
    dependencies,
    candidates,
):
    retriever, hybrid_search, llm_client, llm_factory = dependencies

    ranked_chunk = candidates[6].model_copy(
        update={"rerank_score": 0.97},
    )

    hybrid_search.search.return_value = [
        candidates[2],
        ranked_chunk,
        candidates[0],
    ]

    llm_client.generate_with_usage.return_value = LLMGenerationResult(
        output=LLMOutput(
            answer="Kaynakli yanit. [S2]",
            used_source_ids=["S2"],
            insufficient_context=False,
        ),
        usage=LLMUsage(
            input_tokens=1200,
            output_tokens=80,
        ),
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    result = await pipeline.answer_question(
        "Butunleme sinavina kimler girebilir?",
        RetrievalMethod.HYBRID_RERANK,
    )

    assert len(result.sources) == 1

    source = result.sources[0]

    assert source.source_id == "S2"
    assert source.chunk_id == ranked_chunk.chunk_id
    assert source.document_id == ranked_chunk.document_id
    assert source.document_title == ranked_chunk.document_title
    assert source.source_url == ranked_chunk.source_url
    assert source.page_number == ranked_chunk.page_number
    assert source.excerpt == ranked_chunk.text
    assert source.retrieval_score == ranked_chunk.retrieval_score
    assert source.rerank_score == ranked_chunk.rerank_score

    assert source.chunk_id == "chunk-7"
    assert source.page_number == 7
    assert source.rerank_score == 0.97


@pytest.mark.asyncio
async def test_pipeline_returns_latency_and_usage(
    dependencies,
    monkeypatch,
):
    from backend.app.rag import pipeline as pipeline_module

    retriever, hybrid_search, _llm_client, llm_factory = dependencies

    clock = Mock(
        side_effect=[
            0.0,
            0.125,
            0.375,
            0.5,
            1.25,
            1.5,
        ],
    )

    monkeypatch.setattr(
        pipeline_module,
        "perf_counter",
        clock,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    result = await pipeline.answer_question(
        "Butunleme sinavina kimler girebilir?",
        RetrievalMethod.HYBRID_RERANK,
    )

    metrics = result.metrics

    assert metrics.retrieval_latency_ms == 250
    assert metrics.generation_latency_ms == 750
    assert metrics.total_latency_ms == 1500

    assert metrics.input_tokens == 1200
    assert metrics.output_tokens == 80

    assert metrics.estimated_cost_usd == pytest.approx(
        0.000608,
    )


@pytest.mark.asyncio
async def test_pipeline_empty_context_has_zero_tokens(
    dependencies,
):
    retriever, hybrid_search, _llm_client, llm_factory = dependencies

    retriever.retrieve.return_value = []

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    result = await pipeline.answer_question(
        "Butunleme sinavina kimler girebilir?",
        RetrievalMethod.BM25,
    )

    assert result.metrics.input_tokens == 0
    assert result.metrics.output_tokens == 0
    assert result.metrics.estimated_cost_usd == 0.0
    assert result.metrics.generation_latency_ms == 0

    llm_factory.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_handles_missing_usage(
    dependencies,
):
    retriever, hybrid_search, llm_client, llm_factory = dependencies

    llm_client.generate_with_usage.return_value.usage = None

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    result = await pipeline.answer_question(
        "Butunleme sinavina kimler girebilir?",
        RetrievalMethod.DENSE,
    )

    assert result.metrics.input_tokens is None
    assert result.metrics.output_tokens is None
    assert result.metrics.estimated_cost_usd is None


@pytest.mark.asyncio
async def test_pipeline_handles_unknown_model_price(
    dependencies,
):
    retriever, hybrid_search, llm_client, llm_factory = dependencies

    llm_client.model = "unknown-test-model"

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    result = await pipeline.answer_question(
        "Butunleme sinavina kimler girebilir?",
        RetrievalMethod.HYBRID,
    )

    assert result.metrics.input_tokens == 1200
    assert result.metrics.output_tokens == 80
    assert result.metrics.estimated_cost_usd is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "method"),
    [
        ("", RetrievalMethod.BM25),
        ("   ", RetrievalMethod.DENSE),
        ("ab", RetrievalMethod.HYBRID),
        ("x" * 1001, RetrievalMethod.HYBRID_RERANK),
        ("Valid university question", "invalid_method"),
    ],
)
async def test_pipeline_rejects_invalid_input(
    question,
    method,
    dependencies,
):
    from pydantic import ValidationError

    retriever, hybrid_search, llm_client, llm_factory = dependencies

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    with pytest.raises(ValidationError):
        await pipeline.answer_question(question, method)

    retriever.retrieve.assert_not_awaited()
    hybrid_search.search.assert_not_awaited()
    llm_factory.assert_not_called()
    llm_client.generate_with_usage.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "generation_error",
        "invalid_sources",
    ],
)
async def test_pipeline_closes_llm_client(
    scenario,
    dependencies,
):
    from backend.app.rag.llm_client import LLMClientError

    retriever, hybrid_search, llm_client, llm_factory = dependencies

    llm_client.aclose = AsyncMock()

    pipeline = RAGPipeline(
        retriever=retriever,
        hybrid_search=hybrid_search,
        llm_factory=llm_factory,
    )

    if scenario == "generation_error":
        llm_client.generate_with_usage.side_effect = LLMClientError(
            "Mock generation failure."
        )

    elif scenario == "invalid_sources":
        llm_client.generate_with_usage.return_value = LLMGenerationResult(
            output=LLMOutput(
                answer="Invalid source. [S9]",
                used_source_ids=["S9"],
                insufficient_context=False,
            ),
            usage=None,
        )

    if scenario == "success":
        result = await pipeline.answer_question(
            "Valid university question",
            RetrievalMethod.BM25,
        )

        assert result.insufficient_context is False

    else:
        with pytest.raises(LLMClientError):
            await pipeline.answer_question(
                "Valid university question",
                RetrievalMethod.BM25,
            )

    llm_client.aclose.assert_awaited_once()
