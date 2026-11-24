from datetime import UTC, datetime
from typing import Annotated, Protocol
from uuid import UUID, uuid4

from fastapi import Depends

from backend.app.db.database import SQLiteDatabase, get_database
from backend.app.schemas.documents import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
)


class DocumentService(Protocol):
    async def create_document(
        self, metadata: DocumentCreate, file_name: str
    ) -> DocumentResponse: ...

    async def list_documents(self) -> DocumentListResponse: ...

    async def get_document(self, document_id: UUID) -> DocumentResponse | None: ...

    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> DocumentResponse | None: ...

    async def delete_document(self, document_id: UUID) -> bool: ...


class SQLiteDocumentService:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def create_document(
        self, metadata: DocumentCreate, file_name: str
    ) -> DocumentResponse:
        document = DocumentResponse(
            document_id=uuid4(),
            title=metadata.title,
            source_url=metadata.source_url,
            file_name=file_name,
            status=DocumentStatus.UPLOADED,
            created_at=datetime.now(UTC),
        )
        async with self._database.connect() as connection:
            await connection.execute(
                """INSERT INTO documents
                   (document_id, title, source_url, file_name, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(document.document_id),
                    document.title,
                    document.source_url,
                    document.file_name,
                    document.status.value,
                    document.created_at.isoformat(),
                ),
            )
            await connection.commit()
        return document

    async def list_documents(self) -> DocumentListResponse:
        async with self._database.connect() as connection:
            rows = await connection.execute_fetchall(
                "SELECT * FROM documents ORDER BY created_at DESC, document_id"
            )
        documents = [DocumentResponse.model_validate(dict(row)) for row in rows]
        return DocumentListResponse(documents=documents, total=len(documents))

    async def get_document(self, document_id: UUID) -> DocumentResponse | None:
        async with self._database.connect() as connection, connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (str(document_id),)
        ) as cursor:
            row = await cursor.fetchone()
        return DocumentResponse.model_validate(dict(row)) if row is not None else None

    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> DocumentResponse | None:
        async with self._database.connect() as connection:
            async with connection.execute(
                """UPDATE documents
                   SET status = ?, page_count = COALESCE(?, page_count),
                       chunk_count = COALESCE(?, chunk_count), error_message = ?
                   WHERE document_id = ? RETURNING *""",
                (status.value, page_count, chunk_count, error_message, str(document_id)),
            ) as cursor:
                row = await cursor.fetchone()
            await connection.commit()
        return DocumentResponse.model_validate(dict(row)) if row is not None else None

    async def delete_document(self, document_id: UUID) -> bool:
        async with self._database.connect() as connection:
            async with connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (str(document_id),)
            ) as cursor:
                deleted = cursor.rowcount > 0
            await connection.commit()
        return deleted


def get_document_service(
    database: Annotated[SQLiteDatabase, Depends(get_database)],
) -> DocumentService:
    return SQLiteDocumentService(database)
