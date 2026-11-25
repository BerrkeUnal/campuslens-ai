"""Failure paths of DocumentIngestionService that must stay controlled."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from backend.app.ports.document_processor import DocumentProcessor
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.pdf_extractor import EncryptedPDFError, NoExtractableTextError
from backend.app.schemas.documents import (
    DocumentCreate,
    DocumentResponse,
    DocumentStatus,
)
from backend.app.services.document_ingestion_service import DocumentIngestionService
from backend.app.services.document_service import DocumentService

METADATA = DocumentCreate(title="Yönerge", source_url="https://example.edu.tr/y.pdf")


def _document(status: DocumentStatus, document_id) -> DocumentResponse:
    return DocumentResponse(
        document_id=document_id,
        title="Yönerge",
        source_url="https://example.edu.tr/y.pdf",
        file_name="y.pdf",
        status=status,
        created_at=datetime.now(UTC),
    )


def _service(tmp_path: Path, processor: Mock, status_updates: list):
    document_id = uuid4()
    documents = AsyncMock(spec=DocumentService)
    documents.create_document.return_value = _document(
        DocumentStatus.UPLOADED, document_id
    )
    documents.update_document_status.side_effect = status_updates
    store = AsyncMock(spec=VectorStore)
    service = DocumentIngestionService(
        documents, processor, store, documents_dir=tmp_path
    )
    return service, documents, store, document_id


@pytest.mark.asyncio
async def test_no_chunks_is_a_controlled_422(tmp_path: Path) -> None:
    processor = Mock(spec=DocumentProcessor)
    processor.process_document.return_value = []
    processing = _document(DocumentStatus.PROCESSING, uuid4())
    service, documents, store, _ = _service(tmp_path, processor, [processing, None])

    with (
        patch(
            "backend.app.services.document_ingestion_service.save_uploaded_pdf",
            return_value=tmp_path / "x.pdf",
        ),
        pytest.raises(NoExtractableTextError) as exc,
    ):
        await service.ingest_document(METADATA, "y.pdf", b"%PDF-1.7")

    assert exc.value.status_code == 422
    store.upsert_chunks.assert_not_awaited()
    store.delete_document.assert_awaited_once()
    failed_call = documents.update_document_status.await_args_list[-1]
    assert failed_call.args[1] == DocumentStatus.FAILED
    assert failed_call.kwargs["error_message"] == exc.value.message


@pytest.mark.asyncio
async def test_database_error_while_marking_failed_does_not_hide_original_error(
    tmp_path: Path,
) -> None:
    processor = Mock(spec=DocumentProcessor)
    processor.process_document.side_effect = EncryptedPDFError("password protected")
    processing = _document(DocumentStatus.PROCESSING, uuid4())
    service, documents, store, document_id = _service(
        tmp_path, processor, [processing, RuntimeError("database is locked")]
    )

    with (
        patch(
            "backend.app.services.document_ingestion_service.save_uploaded_pdf",
            return_value=tmp_path / "x.pdf",
        ),
        pytest.raises(EncryptedPDFError),  # the user still gets the 422, not a 500
    ):
        await service.ingest_document(METADATA, "y.pdf", b"%PDF-1.7")

    store.delete_document.assert_awaited_once_with(str(document_id))
    assert (
        documents.update_document_status.await_args_list[-1].args[1]
        == DocumentStatus.FAILED
    )
