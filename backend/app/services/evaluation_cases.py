import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)

from backend.app.schemas.evaluations import EvaluationSplit

ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "data/evaluation/cases.jsonl"
CATALOG_PATH = ROOT / "resources/istinye_documents.json"

NonEmptyText = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
PageNumber = Annotated[int, Field(strict=True, ge=1)]


class EvaluationDataError(ValueError):
    """Invalid evaluation data or document catalog."""


class EvaluationCase(BaseModel):
    case_id: NonEmptyText
    split: EvaluationSplit
    question: NonEmptyText
    answerable: StrictBool
    gold_answer: NonEmptyText
    gold_document_id: NonEmptyText | None
    gold_source_url: NonEmptyText | None = None
    gold_pages: list[PageNumber]
    topic: NonEmptyText

    @model_validator(mode="after")
    def validate_evidence(self) -> "EvaluationCase":
        if self.answerable:
            if not self.gold_document_id or not self.gold_pages:
                raise ValueError("Cevaplanabilir soru için gold_document_id ve gold_pages gerekli.")
        elif self.gold_document_id or self.gold_source_url or self.gold_pages:
            raise ValueError("Cevapsız soru için altın belge, URL ve sayfa bulunamaz.")
        return self


@dataclass(frozen=True)
class ResolvedEvaluationCase:
    case: EvaluationCase
    source_url: str | None


def load_catalog(path: Path) -> dict[str, str]:
    try:
        documents = json.loads(path.read_text(encoding="utf-8-sig"))["documents"]
        if not isinstance(documents, list):
            raise TypeError("documents bir liste olmalı")
        urls: dict[str, str] = {}
        for document in documents:
            document_id = document["id"]
            url = document["source_url"]
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("Belge kimliği boş olamaz")
            if document_id in urls:
                raise ValueError(f"tekrarlanan belge kimliği: {document_id}")
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                raise ValueError(f"{document_id} için geçersiz source_url")
            urls[document_id] = url
        return urls
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise EvaluationDataError(f"{path}: belge kataloğu okunamadı: {exc}") from exc


def load_evaluation_cases(
    cases_path: Path = CASES_PATH, catalog_path: Path = CATALOG_PATH
) -> list[ResolvedEvaluationCase]:
    urls = load_catalog(catalog_path)
    seen: set[str] = set()
    results: list[ResolvedEvaluationCase] = []
    try:
        lines = cases_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationDataError(f"{cases_path}: soru dosyası okunamadı: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"{cases_path}:{line_number}"
        try:
            raw_case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(f"{location}: bozuk JSON: {exc.msg}") from exc
        try:
            case = EvaluationCase.model_validate(raw_case)
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, error["loc"])) or error["msg"] for error in exc.errors())
            raise EvaluationDataError(f"{location}: geçersiz alanlar: {fields}") from exc
        if case.case_id in seen:
            raise EvaluationDataError(f"{location}: tekrarlanan case_id: {case.case_id}")
        source_url = urls.get(case.gold_document_id) if case.gold_document_id else None
        if case.answerable and source_url is None:
            raise EvaluationDataError(
                f"{location}: gold_document_id katalogda bulunamadı: {case.gold_document_id}"
            )
        if case.gold_source_url is not None and case.gold_source_url != source_url:
            raise EvaluationDataError(f"{location}: gold_source_url katalogla eşleşmiyor")
        seen.add(case.case_id)
        results.append(ResolvedEvaluationCase(case, source_url))
    return results
