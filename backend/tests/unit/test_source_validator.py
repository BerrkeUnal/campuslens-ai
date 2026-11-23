
import pytest

from backend.app.rag.llm_client import LLMOutput
from backend.app.rag.source_validator import (
    SourceValidationError,
    validate_sources,
)
from backend.app.schemas.retrieval import ChunkRecord


@pytest.fixture
def source_map():
    return {
        "S1": ChunkRecord(
            chunk_id="doc1-p4-c1",
            document_id="doc1",
            document_title="Mock Regulation",
            source_url="https://example.edu.tr/regulation.pdf",
            page_number=4,
            section_title="Exams",
            text="Students may take the make-up exam.",
        )
    }


def test_valid_source(source_map):
    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S1]",
        used_source_ids=["S1"],
        insufficient_context=False,
    )

    assert validate_sources(output, source_map) == output


def test_unknown_source_is_rejected(source_map):
    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S9]",
        used_source_ids=["S9"],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError):
        validate_sources(output, source_map)


def test_mismatched_citations_are_rejected(source_map):
    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S1]",
        used_source_ids=[],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError):
        validate_sources(output, source_map)


def test_uncited_grounded_answer_is_rejected(source_map):
    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz.",
        used_source_ids=[],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError):
        validate_sources(output, source_map)


def test_insufficient_context_without_citations(source_map):
    output = LLMOutput(
        answer="Bu soruyu yanıtlamak için yeterli bilgi bulunamadı.",
        used_source_ids=[],
        insufficient_context=True,
    )

    assert validate_sources(output, source_map) == output
