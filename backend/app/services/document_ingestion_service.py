import logging
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends
from starlette.concurrency import run_in_threadpool

from backend.app.core.exceptions import ApplicationError
from backend.app.ports.document_processor import DocumentProcessor
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.qdrant_store import get_vector_store
from backend.app.schemas.documents import (
    DocumentCreate,
    DocumentResponse,
    DocumentStatus,
)
from backend.app.services.document_processor import (
    DOCUMENTS_DIR,
    delete_document_file,
    get_document_processor,
    save_uploaded_pdf,
)
from backend.app.services.document_service import (
    DocumentService,
    get_document_service,
)

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Coordinate file storage, PDF processing, SQLite and Qdrant."""

    def __init__(
        self,
        document_service: DocumentService,
        document_processor: DocumentProcessor,
        vector_store: VectorStore,
        documents_dir: Path = DOCUMENTS_DIR,
    ) -> None:
        self._document_service = document_service
        self._document_processor = document_processor
        self._vector_store = vector_store
        self._documents_dir = documents_dir

    async def ingest_document(
        self,
        metadata: DocumentCreate,
        file_name: str,
        content: bytes,
    ) -> DocumentResponse:
        document = await self._document_service.create_document(
            metadata,
            file_name,
        )

        try:
            file_path = await run_in_threadpool(
                save_uploaded_pdf,
                content,
                document.document_id,
                self._documents_dir,
            )

            processing_document = await self._document_service.update_document_status(
                document.document_id,
                DocumentStatus.PROCESSING,
            )

            if processing_document is None:
                raise RuntimeError("Belge kaydı işleme sırasında bulunamadı.")

            chunks = await run_in_threadpool(
                self._document_processor.process_document,
                file_path,
                processing_document,
            )

            if not chunks:
                raise RuntimeError("Belge işleyici herhangi bir parça üretmedi.")

            await self._vector_store.upsert_chunks(chunks)

            indexed_document = await self._document_service.update_document_status(
                document.document_id,
                DocumentStatus.INDEXED,
                page_count=max(chunk.page_number for chunk in chunks),
                chunk_count=len(chunks),
            )

            if indexed_document is None:
                raise RuntimeError("Belge kaydı indeksleme sonrasında bulunamadı.")

            return indexed_document

        except Exception as exc:
            logger.exception(
                "Document ingestion failed: %s",
                document.document_id,
            )

            await self._rollback_failed_ingestion(document.document_id)

            error_message = (
                exc.message
                if isinstance(exc, ApplicationError)
                else ("Belge işlenirken beklenmeyen bir hata oluştu.")
            )

            await self._document_service.update_document_status(
                document.document_id,
                DocumentStatus.FAILED,
                error_message=error_message,
            )

            raise

    async def delete_document(
        self,
        document_id: UUID,
    ) -> bool:
        document = await self._document_service.get_document(document_id)

        if document is None:
            return False

        await self._vector_store.delete_document(str(document_id))

        await run_in_threadpool(
            delete_document_file,
            document_id,
            self._documents_dir,
        )

        return await self._document_service.delete_document(document_id)

    async def _rollback_failed_ingestion(
        self,
        document_id: UUID,
    ) -> None:
        try:
            await self._vector_store.delete_document(str(document_id))
        except Exception:
            logger.exception(
                "Qdrant rollback failed: %s",
                document_id,
            )

        try:
            await run_in_threadpool(
                delete_document_file,
                document_id,
                self._documents_dir,
            )
        except OSError:
            logger.exception(
                "Document file rollback failed: %s",
                document_id,
            )


def get_document_ingestion_service(
    document_service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
    document_processor: Annotated[
        DocumentProcessor,
        Depends(get_document_processor),
    ],
    vector_store: Annotated[
        VectorStore,
        Depends(get_vector_store),
    ],
) -> DocumentIngestionService:
    return DocumentIngestionService(
        document_service,
        document_processor,
        vector_store,
    )
