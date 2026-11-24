import json
from pathlib import Path

import pytest

from backend.app.services.evaluation_cases import (
    EvaluationDataError,
    load_evaluation_cases,
)

URL = "https://example.edu.tr/resmi%20belge.pdf?version=1"
CASE = {
    "case_id": "eval-001",
    "split": "validation",
    "question": "Bütünleme sınavına kimler girebilir?",
    "answerable": True,
    "gold_answer": "Yönetmeliğe göre koşulları sağlayan öğrenciler.",
    "gold_document_id": "regulation",
    "gold_pages": [9, 11],
    "topic": "exams",
}


@pytest.fixture
def evaluation_paths(tmp_path: Path) -> tuple[Path, Path]:
    cases_path = tmp_path / "cases.jsonl"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"documents": [{"id": "regulation", "source_url": URL}]}),
        encoding="utf-8",
    )
    return cases_path, catalog_path


def test_loader_preserves_catalog_url(evaluation_paths: tuple[Path, Path]) -> None:
    cases_path, catalog_path = evaluation_paths
    cases_path.write_text(json.dumps(CASE), encoding="utf-8")
    result, = load_evaluation_cases(cases_path, catalog_path)
    assert result.source_url == URL
    assert result.case.gold_pages == [9, 11]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("{", "bozuk JSON"),
        (json.dumps({**CASE, "gold_document_id": "unknown"}), "katalogda bulunamadı"),
        (json.dumps({**CASE, "gold_pages": []}), "gold_pages"),
        (json.dumps({**CASE, "gold_pages": [0]}), "gold_pages"),
        (json.dumps(CASE) + "\n" + json.dumps(CASE), "tekrarlanan case_id"),
    ],
)
def test_loader_reports_invalid_cases(
    evaluation_paths: tuple[Path, Path], rows: str, message: str
) -> None:
    cases_path, catalog_path = evaluation_paths
    cases_path.write_text(rows, encoding="utf-8")
    with pytest.raises(EvaluationDataError, match=rf"cases\.jsonl:\d:.*{message}"):
        load_evaluation_cases(cases_path, catalog_path)
