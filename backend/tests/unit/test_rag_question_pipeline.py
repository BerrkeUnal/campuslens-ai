import pytest

from backend.app.rag.llm_client import LLMOutput
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk
from backend.app.services.rag_question_pipeline import RAGQuestionPipeline


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, RetrievalMethod]] = []

    async def retrieve(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> list[RetrievedChunk]:
        self.calls.append((question, method))
        return self._chunks


class FakeHybridRerankSearch:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[str] = []

    async def search(
        self,
        question: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append(question)
        return self._chunks[:limit]


class FakeLLMClient:
    def __init__(self, output: LLMOutput) -> None:
        self._output = output
        self.prompts: list[str] = []

    async def generate(self, user_prompt: str) -> LLMOutput:
        self.prompts.append(user_prompt)
        return self._output


def make_chunk(
    index: int,
    method: RetrievalMethod,
    rerank_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        document_id="document-1",
        document_title="Üniversite Yönetmeliği",
        source_url="https://www.example.edu.tr/regulation.pdf",
        page_number=index,
        section_title=f"Madde {index}",
        text=f"{index}. kaynak parçasının metni.",
        retrieval_method=method,
        retrieval_score=1.0 / index,
        rerank_score=rerank_score,
    )


@pytest.mark.asyncio
async def test_pipeline_uses_selected_retrieval_method() -> None:
    chunks = [
        make_chunk(1, RetrievalMethod.DENSE),
        make_chunk(2, RetrievalMethod.DENSE),
    ]
    retriever = FakeRetriever(chunks)
    hybrid_search = FakeHybridRerankSearch([])
    llm_client = FakeLLMClient(
        LLMOutput(
            answer="Yanıt ikinci kaynakta bulunmaktadır. [S2]",
            used_source_ids=["S2"],
            insufficient_context=False,
        )
    )
    pipeline = RAGQuestionPipeline(
        retriever=retriever,
        hybrid_rerank_search=hybrid_search,
        llm_client=llm_client,
    )

    result = await pipeline.answer_question(
        "Öğrenci hangi koşulu sağlamalıdır?",
        RetrievalMethod.DENSE,
    )

    assert retriever.calls == [
        (
            "Öğrenci hangi koşulu sağlamalıdır?",
            RetrievalMethod.DENSE,
        )
    ]
    assert hybrid_search.calls == []
    assert result.answer.endswith("[S2]")
    assert result.insufficient_context is False
    assert len(result.sources) == 1
    assert result.sources[0].source_id == "S2"
    assert result.sources[0].chunk_id == "chunk-2"
    assert result.metrics.retrieval_latency_ms >= 0
    assert result.metrics.generation_latency_ms >= 0
    assert result.metrics.total_latency_ms >= 0


@pytest.mark.asyncio
async def test_pipeline_uses_hybrid_rerank_search() -> None:
    chunks = [
        make_chunk(
            1,
            RetrievalMethod.HYBRID_RERANK,
            rerank_score=0.95,
        )
    ]
    retriever = FakeRetriever([])
    hybrid_search = FakeHybridRerankSearch(chunks)
    llm_client = FakeLLMClient(
        LLMOutput(
            answer="Yanıt birinci kaynakta bulunmaktadır. [S1]",
            used_source_ids=["S1"],
            insufficient_context=False,
        )
    )
    pipeline = RAGQuestionPipeline(
        retriever=retriever,
        hybrid_rerank_search=hybrid_search,
        llm_client=llm_client,
    )

    result = await pipeline.answer_question(
        "Bütünleme sınavına kimler girebilir?",
        RetrievalMethod.HYBRID_RERANK,
    )

    assert retriever.calls == []
    assert hybrid_search.calls == ["Bütünleme sınavına kimler girebilir?"]
    assert result.sources[0].source_id == "S1"
    assert result.sources[0].rerank_score == 0.95


@pytest.mark.asyncio
async def test_pipeline_skips_llm_when_no_chunks_are_found() -> None:
    retriever = FakeRetriever([])
    hybrid_search = FakeHybridRerankSearch([])
    llm_client = FakeLLMClient(
        LLMOutput(
            answer="Bu çıktı kullanılmamalıdır.",
            used_source_ids=[],
            insufficient_context=False,
        )
    )
    pipeline = RAGQuestionPipeline(
        retriever=retriever,
        hybrid_rerank_search=hybrid_search,
        llm_client=llm_client,
    )

    result = await pipeline.answer_question(
        "Belgelerde olmayan bir soru",
        RetrievalMethod.BM25,
    )

    assert result.insufficient_context is True
    assert result.sources == []
    assert llm_client.prompts == []


@pytest.mark.asyncio
async def test_pipeline_shows_examined_sources_for_insufficient_context() -> None:
    chunks = [
        make_chunk(1, RetrievalMethod.HYBRID),
        make_chunk(2, RetrievalMethod.HYBRID),
    ]
    retriever = FakeRetriever(chunks)
    hybrid_search = FakeHybridRerankSearch([])
    llm_client = FakeLLMClient(
        LLMOutput(
            answer="Kaynaklarda bu soruyu yanıtlayacak yeterli bilgi bulunamadı.",
            used_source_ids=[],
            insufficient_context=True,
        )
    )
    pipeline = RAGQuestionPipeline(
        retriever=retriever,
        hybrid_rerank_search=hybrid_search,
        llm_client=llm_client,
    )

    result = await pipeline.answer_question(
        "Yurt ücretleri ne kadar?",
        RetrievalMethod.HYBRID,
    )

    assert result.insufficient_context is True
    assert [source.source_id for source in result.sources] == ["S1", "S2"]
    assert len(llm_client.prompts) == 1
