from pathlib import Path
from typing import Protocol

from backend.app.schemas.documents import DocumentResponse
from backend.app.schemas.retrieval import ChunkRecord


class DocumentProcessor(Protocol):
    def process_document(
        self,
        file_path: Path,
        document: DocumentResponse,
    ) -> list[ChunkRecord]: ...
