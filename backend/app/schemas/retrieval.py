from enum import StrEnum

from pydantic import BaseModel, Field


class RetrievalMethod(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


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
