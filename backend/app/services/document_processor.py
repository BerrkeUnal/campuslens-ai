"""PDF -> list[ChunkRecord]: implementation of the DocumentProcessor port.

Owner: Person 3 (document processing).

* ``PdfDocumentProcessor.process_document`` satisfies
  ``backend.app.ports.document_processor.DocumentProcessor``.
* It is synchronous and CPU-bound. From an async route call it with
  ``await run_in_threadpool(processor.process_document, path, document)``.
* PDFs are stored on disk under ``data/documents/{document_id}.pdf``; SQLite
  only keeps metadata. ``data/*`` is already git-ignored.
* Status updates (processing / indexed / failed) are done by the caller through
  ``DocumentRepository.update_document_status`` (Person 6). Use
  ``ProcessingResult.page_count`` for ``page_count``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.app.core.config import Settings, get_settings
from backend.app.ports.document_processor import DocumentProcessor
from backend.app.rag.chunker import TokenCounter, chunk_pages, get_default_token_counter
from backend.app.rag.pdf_extractor import (
    NoExtractableTextError,
    PageText,
    PDFProcessingError,
    extract_pages,
    validate_pdf,
    validate_pdf_bytes,
)
from backend.app.rag.text_cleaner import clean_pages
from backend.app.schemas.documents import DocumentResponse
from backend.app.schemas.retrieval import ChunkRecord

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")


@dataclass
class ProcessingResult:
    pages: list[PageText]
    chunks: list[ChunkRecord]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def empty_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if not p.text.strip()]


# --------------------------------------------------------------------------- #
# File storage
# --------------------------------------------------------------------------- #
def document_file_path(document_id: UUID | str, base_dir: Path | None = None) -> Path:
    return (base_dir or DOCUMENTS_DIR) / f"{document_id}.pdf"


def save_uploaded_pdf(
    content: bytes,
    document_id: UUID | str,
    base_dir: Path | None = None,
) -> Path:
    """Validate and store an upload as ``{base_dir}/{document_id}.pdf``.

    The stored name comes from the document UUID, never from the user's file
    name. Files that do not open as a PDF are not kept on disk.
    """
    validate_pdf_bytes(content)
    target = document_file_path(document_id, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".pdf.part")
    tmp.write_bytes(content)
    try:
        validate_pdf(tmp)
    except PDFProcessingError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(target)
    return target


def delete_document_file(document_id: UUID | str, base_dir: Path | None = None) -> bool:
    path = document_file_path(document_id, base_dir)
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
class PdfDocumentProcessor:
    """Validate -> extract -> clean -> chunk."""

    def __init__(
        self,
        settings: Settings | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._token_counter = token_counter

    @property
    def token_counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = get_default_token_counter()
        return self._token_counter

    def process(
        self,
        file_path: Path,
        document_id: UUID | str,
        document_title: str,
        source_url: str,
    ) -> ProcessingResult:
        started = time.perf_counter()
        counter = self.token_counter
        pages = clean_pages(extract_pages(Path(file_path)))
        chunks = chunk_pages(
            pages,
            document_id=str(document_id),
            document_title=document_title,
            source_url=source_url,
            chunk_size_tokens=self._settings.chunk_size_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
            token_counter=counter,
        )
        if not chunks:
            raise NoExtractableTextError(f"No chunks produced for {file_path}")

        token_counts = [counter.count(c.text) for c in chunks]
        result = ProcessingResult(pages=pages, chunks=chunks)
        result.stats = {
            "page_count": result.page_count,
            "empty_page_count": len(result.empty_pages),
            "empty_pages": result.empty_pages,
            "chunk_count": len(chunks),
            "min_tokens": min(token_counts),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
            "max_tokens": max(token_counts),
            "chunks_with_section_title": sum(
                c.section_title is not None for c in chunks
            ),
            "tokenizer": counter.name,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        logger.info("Processed document %s: %s", document_id, result.stats)
        return result

    def process_document(
        self,
        file_path: Path,
        document: DocumentResponse,
    ) -> list[ChunkRecord]:
        """DocumentProcessor port method. Raises ``PDFProcessingError``
        (an ``ApplicationError`` with a Turkish message) on bad input."""
        try:
            return self.process(
                file_path,
                document.document_id,
                document.title,
                document.source_url,
            ).chunks
        except PDFProcessingError as exc:
            logger.warning("Document %s failed: %s", document.document_id, exc.detail)
            raise


def write_outputs(result: ProcessingResult, out_dir: Path) -> tuple[Path, Path]:
    """Write ``pages.json`` and ``chunks.json`` (UTF-8, Turkish kept readable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_path, chunks_path = out_dir / "pages.json", out_dir / "chunks.json"
    pages_path.write_text(
        json.dumps([p.to_dict() for p in result.pages], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(
            [c.model_dump() for c in result.chunks], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return pages_path, chunks_path


_processor = PdfDocumentProcessor()


def get_document_processor() -> DocumentProcessor:
    return _processor
