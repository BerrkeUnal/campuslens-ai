
import re

from backend.app.rag.llm_client import LLMOutput
from backend.app.schemas.retrieval import ChunkRecord


class SourceValidationError(ValueError):
    """Raised when an LLM response contains invalid citations."""


def validate_sources(
    output: LLMOutput,
    source_map: dict[str, ChunkRecord],
) -> LLMOutput:
    """Ensure citations refer only to supplied source chunks."""

    allowed_ids = set(source_map)
    used_ids = output.used_source_ids

    if len(used_ids) != len(set(used_ids)):
        raise SourceValidationError(
            "Duplicate source IDs are not allowed."
        )

    if not set(used_ids).issubset(allowed_ids):
        raise SourceValidationError(
            "LLM referenced an unknown source ID."
        )

    cited_ids = set(
        re.findall(r"\[([^\[\]]+)\]", output.answer)
    )

    if not cited_ids.issubset(allowed_ids):
        raise SourceValidationError(
            "Answer contains an unknown citation."
        )

    if cited_ids != set(used_ids):
        raise SourceValidationError(
            "Answer citations do not match used_source_ids."
        )

    if not output.insufficient_context and not used_ids:
        raise SourceValidationError(
            "A grounded answer must cite at least one source."
        )

    return output
