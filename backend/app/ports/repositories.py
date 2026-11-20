from typing import Protocol
from uuid import UUID

from backend.app.schemas.documents import (
    DocumentResponse,
    DocumentStatus,
)
from backend.app.schemas.evaluations import (
    EvaluationResultResponse,
)
from backend.app.schemas.questions import AskRequest, AskResponse


class DocumentRepository(Protocol):
    async def create_document(
        self,
        document: DocumentResponse,
    ) -> DocumentResponse: ...

    async def get_document(
        self,
        document_id: UUID,
    ) -> DocumentResponse | None: ...

    async def list_documents(
        self,
    ) -> list[DocumentResponse]: ...

    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> DocumentResponse | None: ...

    async def delete_document(
        self,
        document_id: UUID,
    ) -> bool: ...


class QuestionLogRepository(Protocol):
    async def create_question_log(
        self,
        request: AskRequest,
        response: AskResponse,
    ) -> None: ...


class EvaluationRepository(Protocol):
    async def save_result(
        self,
        result: EvaluationResultResponse,
    ) -> None: ...

    async def list_results(
        self,
    ) -> list[EvaluationResultResponse]: ...
