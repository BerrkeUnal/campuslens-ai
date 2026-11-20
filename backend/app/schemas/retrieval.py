from enum import StrEnum

from pydantic import BaseModel, Field


class RetrievalMethod(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source_url: str
    page_number: int = Field(ge=1)
    section_title: str | None = None
    text: str = Field(min_length=1)


class RetrievedChunk(ChunkRecord):
    retrieval_method: RetrievalMethod
    retrieval_score: float
    rerank_score: float | None = None


class SourceResponse(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    document_title: str
    source_url: str
    page_number: int = Field(ge=1)
    excerpt: str
    retrieval_score: float
    rerank_score: float | None = None
