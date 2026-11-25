import pytest
from pydantic import ValidationError

from backend.app.rag.llm_client import LLMOutput


@pytest.mark.parametrize(
    "answer",
    ["", " ", "   ", "\t", "\n", "\t\n  "],
)
@pytest.mark.parametrize(
    "insufficient_context",
    [False, True],
)
def test_llm_output_rejects_blank_answer(
    answer: str,
    insufficient_context: bool,
) -> None:
    with pytest.raises(
        ValidationError,
        match="LLM answer must not be empty",
    ):
        LLMOutput(
            answer=answer,
            used_source_ids=[] if insufficient_context else ["S1"],
            insufficient_context=insufficient_context,
        )


@pytest.mark.parametrize(
    ("answer", "source_ids", "insufficient_context"),
    [
        (
            "Öğrenci kayıtları 15 Eylül tarihinde başlar. [S1]",
            ["S1"],
            False,
        ),
        (
            "Bu soruyu yanıtlamak için yeterli kaynak bulunamadı.",
            [],
            True,
        ),
        (
            "  Geçerli cevap. [S1]  ",
            ["S1"],
            False,
        ),
    ],
)
def test_llm_output_accepts_non_empty_answer(
    answer: str,
    source_ids: list[str],
    insufficient_context: bool,
) -> None:
    output = LLMOutput(
        answer=answer,
        used_source_ids=source_ids,
        insufficient_context=insufficient_context,
    )

    assert output.answer == answer
    assert output.used_source_ids == source_ids
    assert output.insufficient_context is insufficient_context


def test_llm_output_preserves_json_schema_contract() -> None:
    schema = LLMOutput.model_json_schema()

    assert schema["additionalProperties"] is False

    assert set(schema["required"]) == {
        "answer",
        "used_source_ids",
        "insufficient_context",
    }

    assert schema["properties"]["answer"] == {
        "title": "Answer",
        "type": "string",
    }

    assert schema["properties"]["used_source_ids"]["type"] == "array"
    assert schema["properties"]["used_source_ids"]["items"]["type"] == "string"
    assert schema["properties"]["insufficient_context"]["type"] == "boolean"
