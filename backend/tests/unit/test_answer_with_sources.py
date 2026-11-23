
from unittest.mock import AsyncMock

import pytest

from backend.app.rag.llm_client import (
    LLMOutput,
    answer_with_sources,
)
from backend.app.rag.source_validator import SourceValidationError
from backend.app.schemas.retrieval import ChunkRecord


@pytest.fixture
def mock_chunks():
    return [
        ChunkRecord(
            chunk_id="doc1-p4-c1",
            document_id="doc1",
            document_title="Sinav Yonetmeligi",
            source_url="https://example.edu.tr/sinav.pdf",
            page_number=4,
            section_title="Butunleme Sinavi",
            text=(
                "Devamsizlik nedeniyle basarisiz olan ogrenciler "
                "butunleme sinavina giremez."
            ),
        )
    ]


@pytest.mark.asyncio
async def test_answer_with_valid_citation(mock_chunks):
    client = AsyncMock()
    client.generate.return_value = LLMOutput(
        answer="Bütünleme sınavına giremezsiniz. [S1]",
        used_source_ids=["S1"],
        insufficient_context=False,
    )

    result = await answer_with_sources(
        "Devamsızlıktan kaldım. Bütünlemeye girebilir miyim?",
        mock_chunks,
        client,
    )

    assert result.used_source_ids == ["S1"]
    assert result.insufficient_context is False

    client.generate.assert_awaited_once()
    prompt = client.generate.call_args.args[0]
    assert "[S1]" in prompt
    assert "Sinav Yonetmeligi" in prompt


@pytest.mark.asyncio
async def test_answer_rejects_fabricated_citation(mock_chunks):
    client = AsyncMock()
    client.generate.return_value = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S9]",
        used_source_ids=["S9"],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError):
        await answer_with_sources(
            "Bütünlemeye girebilir miyim?",
            mock_chunks,
            client,
        )


@pytest.mark.asyncio
async def test_empty_context_skips_llm():
    client = AsyncMock()

    result = await answer_with_sources(
        "Bütünlemeye girebilir miyim?",
        [],
        client,
    )

    assert result.insufficient_context is True
    assert result.used_source_ids == []
    client.generate.assert_not_awaited()
