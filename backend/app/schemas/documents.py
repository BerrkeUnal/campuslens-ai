from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

DocumentTitle = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=2,
        max_length=200,
    ),
]

SourceUrl = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=8,
        max_length=2048,
    ),
]


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentCreate(BaseModel):
    title: DocumentTitle
    source_url: SourceUrl


class DocumentResponse(BaseModel):
    document_id: UUID
    title: str
    source_url: str
    file_name: str
    status: DocumentStatus
    page_count: int | None = Field(default=None, ge=1)
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int = Field(ge=0)
