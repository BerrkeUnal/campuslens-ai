from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from backend.app.schemas.documents import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
)


class DocumentService(Protocol):
    async def create_document(
        self,
        metadata: DocumentCreate,
        file_name: str,
    ) -> DocumentResponse: ...

    async def list_documents(self) -> DocumentListResponse: ...

    async def delete_document(self, document_id: UUID) -> bool: ...


class InMemoryDocumentService:
    def __init__(self) -> None:
        self._documents: dict[UUID, DocumentResponse] = {}

    async def create_document(
        self,
        metadata: DocumentCreate,
        file_name: str,
    ) -> DocumentResponse:
        document = DocumentResponse(
            document_id=uuid4(),
            title=metadata.title,
            source_url=metadata.source_url,
            file_name=file_name,
            status=DocumentStatus.UPLOADED,
            page_count=None,
            chunk_count=0,
            created_at=datetime.now(UTC),
            error_message=None,
        )

        self._documents[document.document_id] = document

        return document

    async def list_documents(self) -> DocumentListResponse:
        documents = sorted(
            self._documents.values(),
            key=lambda document: document.created_at,
            reverse=True,
        )

        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )

    async def delete_document(self, document_id: UUID) -> bool:
        if document_id not in self._documents:
            return False

        del self._documents[document_id]
        return True


_document_service = InMemoryDocumentService()


def get_document_service() -> DocumentService:
    return _document_service
