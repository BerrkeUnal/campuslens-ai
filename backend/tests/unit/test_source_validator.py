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


def test_grouped_citations_are_accepted(source_map):
    source_map["S2"] = source_map["S1"].model_copy(
        update={"chunk_id": "doc1-p5-c1", "page_number": 5}
    )

    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S1, S2]",
        used_source_ids=["S1", "S2"],
        insufficient_context=False,
    )

    assert validate_sources(output, source_map) == output


def test_normal_brackets_are_ignored(source_map):
    output = LLMOutput(
        answer="Madde 18 [1] uyarınca sınava girebilirsiniz. [S1]",
        used_source_ids=["S1"],
        insufficient_context=False,
    )

    assert validate_sources(output, source_map) == output


@pytest.mark.parametrize(
    "citation",
    ["[S1, S2]", "[S1,S2]", "[S1][S2]"],
)
def test_multiple_citation_formats(source_map, citation):
    source_map["S2"] = source_map["S1"].model_copy(
        update={"chunk_id": "doc1-p5-c1", "page_number": 5}
    )

    output = LLMOutput(
        answer=f"Bütünleme sınavına girebilirsiniz. {citation}",
        used_source_ids=["S1", "S2"],
        insufficient_context=False,
    )

    assert validate_sources(output, source_map) == output


def test_unknown_grouped_source_is_rejected(source_map):
    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S1, S99]",
        used_source_ids=["S1"],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError, match="unknown citation"):
        validate_sources(output, source_map)


def test_malformed_grouped_citation_is_rejected(source_map):
    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S1, abc]",
        used_source_ids=["S1"],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError, match="malformed citation"):
        validate_sources(output, source_map)


def test_duplicate_source_ids_are_rejected(source_map):
    output = LLMOutput(
        answer="Bütünleme sınavına girebilirsiniz. [S1]",
        used_source_ids=["S1", "S1"],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError, match="Duplicate source IDs"):
        validate_sources(output, source_map)


def test_normal_brackets_do_not_replace_required_citations(source_map):
    output = LLMOutput(
        answer="Madde 18 [1] uyarınca sınava girebilirsiniz.",
        used_source_ids=[],
        insufficient_context=False,
    )

    with pytest.raises(SourceValidationError, match="must cite"):
        validate_sources(output, source_map)
