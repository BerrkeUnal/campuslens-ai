from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.core.exceptions import ServiceUnavailableError
from backend.app.db.database import SQLiteDatabase
from backend.app.schemas.documents import DocumentCreate, DocumentStatus
from backend.app.services.document_service import SQLiteDocumentService


@pytest.mark.asyncio
async def test_document_lifecycle_survives_reopening_database(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'data' / 'campuslens.db').as_posix()}"
    database = SQLiteDatabase(url)
    await database.initialize()
    await database.initialize()
    assert await database.health_status() == "ok"
    service = SQLiteDocumentService(database)
    official_url = "https://example.edu.tr/y%C3%B6netmelik.pdf?version=1"
    document = await service.create_document(
        DocumentCreate(title="Yönetmelik", source_url=official_url), "regulation.pdf"
    )

    assert document.status is DocumentStatus.UPLOADED
    assert database.path.is_file()
    with pytest.raises(ServiceUnavailableError):
        await service.update_document_status(
            document.document_id, DocumentStatus.PROCESSING, 0
        )
    assert await service.get_document(document.document_id) == document

    processing = await service.update_document_status(
        document.document_id, DocumentStatus.PROCESSING, page_count=14, chunk_count=27
    )
    assert processing is not None
    assert processing.status is DocumentStatus.PROCESSING
    failed = await service.update_document_status(
        document.document_id, DocumentStatus.FAILED, error_message="PDF cannot be read."
    )
    assert failed is not None
    assert failed.error_message == "PDF cannot be read."
    indexed = await service.update_document_status(
        document.document_id, DocumentStatus.INDEXED
    )
    assert indexed is not None
    assert indexed.error_message is None

    reopened = SQLiteDatabase(url)
    await reopened.initialize()
    service = SQLiteDocumentService(reopened)
    stored = await service.get_document(document.document_id)
    assert stored == indexed
    assert stored.source_url == official_url
    assert (stored.page_count, stored.chunk_count) == (14, 27)
    assert (await service.list_documents()).documents == [stored]
    assert await service.update_document_status(uuid4(), DocumentStatus.FAILED) is None
    assert await service.delete_document(document.document_id)
    assert not await service.delete_document(document.document_id)
    assert await service.get_document(document.document_id) is None
    assert (await service.list_documents()).total == 0


@pytest.mark.asyncio
async def test_database_open_failure_is_reported_as_service_unavailable(
    tmp_path: Path,
) -> None:
    database = SQLiteDatabase(f"sqlite:///{tmp_path.as_posix()}")
    assert await database.health_status() == "unavailable"
    with pytest.raises(ServiceUnavailableError) as exc:
        await database.initialize()
    assert exc.value.status_code == 503
    assert exc.value.__cause__ is not None
