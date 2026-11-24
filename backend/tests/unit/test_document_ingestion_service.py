from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch
from uuid import UUID, uuid4

import pytest

from backend.app.ports.document_processor import DocumentProcessor
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.pdf_extractor import NoExtractableTextError
from backend.app.schemas.documents import (
    DocumentCreate,
    DocumentResponse,
    DocumentStatus,
)
from backend.app.schemas.retrieval import ChunkRecord
from backend.app.services.document_ingestion_service import (
    DocumentIngestionService,
)
from backend.app.services.document_service import DocumentService


def make_document(
    document_id: UUID,
    status: DocumentStatus,
    *,
    page_count: int | None = None,
    chunk_count: int = 0,
    error_message: str | None = None,
) -> DocumentResponse:
    return DocumentResponse(
        document_id=document_id,
        title="Sınav Yönetmeliği",
        source_url="https://example.edu.tr/sinav.pdf",
        file_name="sinav.pdf",
        status=status,
        page_count=page_count,
        chunk_count=chunk_count,
        created_at=datetime.now(UTC),
        error_message=error_message,
    )


def make_chunks(
    document_id: UUID,
) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=f"{document_id}-p{page}-c1",
            document_id=str(document_id),
            document_title="Sınav Yönetmeliği",
            source_url="https://example.edu.tr/sinav.pdf",
            page_number=page,
            text=f"Sayfa {page} kaynak metni.",
        )
        for page in (1, 2)
    ]


@pytest.mark.asyncio
async def test_ingest_document_indexes_chunks_and_updates_status(
    tmp_path: Path,
) -> None:
    document_id = uuid4()

    metadata = DocumentCreate(
        title="Sınav Yönetmeliği",
        source_url="https://example.edu.tr/sinav.pdf",
    )

    uploaded = make_document(
        document_id,
        DocumentStatus.UPLOADED,
    )

    processing = make_document(
        document_id,
        DocumentStatus.PROCESSING,
    )

    indexed = make_document(
        document_id,
        DocumentStatus.INDEXED,
        page_count=2,
        chunk_count=2,
    )

    chunks = make_chunks(document_id)
    stored_path = tmp_path / f"{document_id}.pdf"

    document_service = AsyncMock(
        spec=DocumentService,
    )

    document_service.create_document.return_value = uploaded

    document_service.update_document_status.side_effect = [
        processing,
        indexed,
    ]

    document_processor = Mock(
        spec=DocumentProcessor,
    )

    document_processor.process_document.return_value = chunks

    vector_store = AsyncMock(
        spec=VectorStore,
    )

    service = DocumentIngestionService(
        document_service,
        document_processor,
        vector_store,
        documents_dir=tmp_path,
    )

    content = b"%PDF-1.7 test"

    with patch(
        ("backend.app.services.document_ingestion_service.save_uploaded_pdf"),
        return_value=stored_path,
    ) as save_uploaded_pdf:
        result = await service.ingest_document(
            metadata,
            "sinav.pdf",
            content,
        )

    assert result == indexed

    document_service.create_document.assert_awaited_once_with(
        metadata,
        "sinav.pdf",
    )

    save_uploaded_pdf.assert_called_once_with(
        content,
        document_id,
        tmp_path,
    )

    document_processor.process_document.assert_called_once_with(
        stored_path,
        processing,
    )

    vector_store.upsert_chunks.assert_awaited_once_with(
        chunks,
    )

    document_service.update_document_status.assert_has_awaits(
        [
            call(
                document_id,
                DocumentStatus.PROCESSING,
            ),
            call(
                document_id,
                DocumentStatus.INDEXED,
                page_count=2,
                chunk_count=2,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_ingest_failure_rolls_back_and_marks_document_failed(
    tmp_path: Path,
) -> None:
    document_id = uuid4()

    metadata = DocumentCreate(
        title="Sınav Yönetmeliği",
        source_url="https://example.edu.tr/sinav.pdf",
    )

    uploaded = make_document(
        document_id,
        DocumentStatus.UPLOADED,
    )

    processing = make_document(
        document_id,
        DocumentStatus.PROCESSING,
    )

    error = NoExtractableTextError("No text found in the PDF.")

    failed = make_document(
        document_id,
        DocumentStatus.FAILED,
        error_message=error.message,
    )

    stored_path = tmp_path / f"{document_id}.pdf"

    document_service = AsyncMock(
        spec=DocumentService,
    )

    document_service.create_document.return_value = uploaded

    document_service.update_document_status.side_effect = [
        processing,
        failed,
    ]

    document_processor = Mock(
        spec=DocumentProcessor,
    )

    document_processor.process_document.side_effect = error

    vector_store = AsyncMock(
        spec=VectorStore,
    )

    service = DocumentIngestionService(
        document_service,
        document_processor,
        vector_store,
        documents_dir=tmp_path,
    )

    with (
        patch(
            ("backend.app.services.document_ingestion_service.save_uploaded_pdf"),
            return_value=stored_path,
        ),
        patch(
            ("backend.app.services.document_ingestion_service.delete_document_file"),
            return_value=True,
        ) as delete_document_file,
        pytest.raises(
            NoExtractableTextError,
        ),
    ):
        await service.ingest_document(
            metadata,
            "sinav.pdf",
            b"%PDF-1.7 test",
        )

    vector_store.upsert_chunks.assert_not_awaited()

    vector_store.delete_document.assert_awaited_once_with(str(document_id))

    delete_document_file.assert_called_once_with(
        document_id,
        tmp_path,
    )

    document_service.update_document_status.assert_has_awaits(
        [
            call(
                document_id,
                DocumentStatus.PROCESSING,
            ),
            call(
                document_id,
                DocumentStatus.FAILED,
                error_message=error.message,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_delete_document_removes_vector_file_and_database_record(
    tmp_path: Path,
) -> None:
    document_id = uuid4()

    document = make_document(
        document_id,
        DocumentStatus.INDEXED,
        page_count=2,
        chunk_count=2,
    )

    document_service = AsyncMock(
        spec=DocumentService,
    )

    document_service.get_document.return_value = document
    document_service.delete_document.return_value = True

    document_processor = Mock(
        spec=DocumentProcessor,
    )

    vector_store = AsyncMock(
        spec=VectorStore,
    )

    service = DocumentIngestionService(
        document_service,
        document_processor,
        vector_store,
        documents_dir=tmp_path,
    )

    with patch(
        ("backend.app.services.document_ingestion_service.delete_document_file"),
        return_value=True,
    ) as delete_document_file:
        deleted = await service.delete_document(document_id)

    assert deleted is True

    document_service.get_document.assert_awaited_once_with(document_id)

    vector_store.delete_document.assert_awaited_once_with(str(document_id))

    delete_document_file.assert_called_once_with(
        document_id,
        tmp_path,
    )

    document_service.delete_document.assert_awaited_once_with(document_id)
