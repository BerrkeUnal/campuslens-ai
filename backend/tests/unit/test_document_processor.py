import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest

from backend.app.rag.chunker import ApproximateTokenCounter
from backend.app.rag.pdf_extractor import EncryptedPDFError, InvalidPDFError
from backend.app.schemas.documents import DocumentResponse, DocumentStatus
from backend.app.schemas.retrieval import ChunkRecord
from backend.app.services.document_processor import (
    PdfDocumentProcessor,
    delete_document_file,
    save_uploaded_pdf,
    write_outputs,
)


@pytest.fixture
def processor() -> PdfDocumentProcessor:
    # Offline, deterministic counter so tests never download the tokenizer.
    return PdfDocumentProcessor(token_counter=ApproximateTokenCounter())


def make_document(title: str = "Önlisans ve Lisans Yönetmeliği") -> DocumentResponse:
    return DocumentResponse(
        document_id=uuid4(),
        title=title,
        source_url="https://www.example.edu.tr/regulation",
        file_name="regulation.pdf",
        status=DocumentStatus.UPLOADED,
        created_at=datetime.now(UTC),
    )


def test_process_document_follows_port_contract(
    processor, regulation_pdf: Path
) -> None:
    document = make_document()
    chunks = processor.process_document(regulation_pdf, document)

    assert chunks and all(isinstance(c, ChunkRecord) for c in chunks)
    assert {c.document_id for c in chunks} == {str(document.document_id)}
    assert {c.document_title for c in chunks} == {document.title}
    assert {c.source_url for c in chunks} == {document.source_url}
    assert {c.page_number for c in chunks} == {1, 2, 4}  # page 3 is empty


def test_process_reports_page_statistics(processor, regulation_pdf: Path) -> None:
    result = processor.process(
        regulation_pdf, "doc-1", "Yönetmelik", "https://x.edu.tr"
    )
    assert result.page_count == 4
    assert result.empty_pages == [3]
    assert result.stats["chunk_count"] == len(result.chunks)
    assert result.stats["max_tokens"] <= 400


def test_process_document_raises_controlled_error(processor, tmp_path: Path) -> None:
    path = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "x")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="a", owner_pw="a")

    with pytest.raises(EncryptedPDFError) as exc:
        processor.process_document(path, make_document())
    assert "parola" in exc.value.message


def test_save_uploaded_pdf_uses_document_id(
    tmp_path: Path, regulation_pdf: Path
) -> None:
    document_id = uuid4()
    path = save_uploaded_pdf(
        regulation_pdf.read_bytes(), document_id, base_dir=tmp_path
    )
    assert path == tmp_path / f"{document_id}.pdf" and path.exists()
    assert delete_document_file(document_id, base_dir=tmp_path)
    assert not path.exists()


def test_save_rejects_invalid_pdf_and_leaves_no_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidPDFError):
        save_uploaded_pdf(b"not a pdf", "doc-2", base_dir=tmp_path)
    with pytest.raises(InvalidPDFError):
        save_uploaded_pdf(
            b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF", "doc-3", base_dir=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_write_outputs_is_readable_utf8_json(
    processor, tmp_path: Path, regulation_pdf: Path
) -> None:
    result = processor.process(
        regulation_pdf, "doc-4", "Yönetmelik", "https://x.edu.tr"
    )
    pages_path, chunks_path = write_outputs(result, tmp_path)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert [
        ChunkRecord(**c) for c in chunks
    ] == result.chunks  # Person 4 can load as-is
    assert "Öğrenci" in chunks_path.read_text(encoding="utf-8")
    assert len(json.loads(pages_path.read_text(encoding="utf-8"))) == 4


# --- .part files never survive a failed upload ---------------------------------- #
def test_part_file_removed_when_rename_fails(
    tmp_path: Path, regulation_pdf: Path, monkeypatch
) -> None:
    def locked(self, target):  # e.g. Windows: target opened by another process
        raise PermissionError("[WinError 32] The process cannot access the file")

    monkeypatch.setattr(Path, "replace", locked)
    uploads = tmp_path / "uploads"
    content = regulation_pdf.read_bytes()
    with pytest.raises(PermissionError):
        save_uploaded_pdf(content, "doc-7", base_dir=uploads)
    assert list(uploads.iterdir()) == []


def test_part_file_removed_when_validation_crashes(
    tmp_path: Path, regulation_pdf: Path, monkeypatch
) -> None:
    from backend.app.services import document_processor as module

    def crash(path):
        raise RuntimeError("unexpected PyMuPDF failure")

    monkeypatch.setattr(module, "validate_pdf", crash)
    uploads = tmp_path / "uploads"
    with pytest.raises(RuntimeError):
        save_uploaded_pdf(regulation_pdf.read_bytes(), "doc-8", base_dir=uploads)
    assert list(uploads.iterdir()) == []


def test_part_file_removed_for_corrupt_pdf(tmp_path: Path) -> None:
    from backend.tests.pdf_factory import CORRUPT_PDF_BYTES

    with pytest.raises(InvalidPDFError):
        save_uploaded_pdf(CORRUPT_PDF_BYTES, "doc-9", base_dir=tmp_path)
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.iterdir()) == []


def test_delete_document_file_also_removes_leftover_part(tmp_path: Path) -> None:
    (tmp_path / "doc-10.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "doc-10.pdf.part").write_bytes(b"%PDF-1.7")
    assert delete_document_file("doc-10", base_dir=tmp_path) is True
    assert list(tmp_path.iterdir()) == []
