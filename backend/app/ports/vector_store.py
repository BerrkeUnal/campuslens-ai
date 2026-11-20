from typing import Protocol

from backend.app.schemas.retrieval import (
    ChunkRecord,
    RetrievedChunk,
)


class VectorStore(Protocol):
    async def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
    ) -> None: ...

    async def delete_document(
        self,
        document_id: str,
    ) -> None: ...

    async def search_bm25(
        self,
        question: str,
        limit: int,
    ) -> list[RetrievedChunk]: ...

    async def search_dense(
        self,
        question: str,
        limit: int,
    ) -> list[RetrievedChunk]: ...

    async def search_hybrid_rrf(
        self,
        question: str,
        limit: int,
    ) -> list[RetrievedChunk]: ...
