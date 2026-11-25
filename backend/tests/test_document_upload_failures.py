"""End-to-end upload failure tests: controlled 4xx and full rollback.

Real FastAPI app, real SQLite database, real PDF processor; only Qdrant is
replaced by an in-memory store so the tests run offline and on Windows.

For every failed upload we check:
* the HTTP response is the expected controlled error (never a bare 500),
* the SQLite record is not left as "uploaded"/"processing"/"indexed" but is
  marked "failed" with a Turkish error message (the UI shows it),
* no PDF and no ``.part`` file is left in the documents folder,
* no chunk of the document is left in the vector store.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.rag import pdf_extractor
from backend.app.rag.chunker import ApproximateTokenCounter
from backend.app.rag.qdrant_store import VectorStoreUnavailableError
from backend.app.schemas.retrieval import ChunkRecord, RetrievedChunk
from backend.app.services.document_ingestion_service import (
    DocumentIngestionService,
    get_document_ingestion_service,
)
from backend.app.services.document_processor import PdfDocumentProcessor
from backend.app.services.document_service import SQLiteDocumentService
from backend.tests import pdf_factory


class InMemoryVectorStore:
    """Stand-in for Qdrant that records what happened."""

    def __init__(self) -> None:
        self.chunks: dict[str, ChunkRecord] = {}
        self.deleted: list[str] = []
        self.fail_upsert: Exception | None = None
        self.write_before_failing = False

    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        if self.fail_upsert is not None:
            if self.write_before_failing:  # Qdrant died halfway through a batch
                for chunk in chunks[: len(chunks) // 2 or 1]:
                    self.chunks[chunk.chunk_id] = chunk
            raise self.fail_upsert
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

    async def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)
        self.chunks = {
            k: c for k, c in self.chunks.items() if c.document_id != document_id
        }

    def chunks_of(self, document_id: str) -> list[ChunkRecord]:
        return [c for c in self.chunks.values() if c.document_id == document_id]

    async def search_bm25(self, question: str, limit: int) -> list[RetrievedChunk]:
        return []

    async def search_dense(self, question: str, limit: int) -> list[RetrievedChunk]:
        return []

    async def search_hybrid_rrf(
        self, question: str, limit: int
    ) -> list[RetrievedChunk]:
        return []


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def documents_dir(tmp_path: Path) -> Path:
    return tmp_path / "documents"


@pytest.fixture
def client(
    tmp_path: Path, store: InMemoryVectorStore, documents_dir: Path
) -> Iterator[TestClient]:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'campuslens.db').as_posix()}"
    )
    app = create_app(settings)
    service = DocumentIngestionService(
        SQLiteDocumentService(app.state.database),
        PdfDocumentProcessor(token_counter=ApproximateTokenCounter()),
        store,
        documents_dir=documents_dir,
    )
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def upload(client: TestClient, content: bytes, name: str = "yonerge.pdf"):
    return client.post(
        "/api/documents/upload",
        files={"file": (name, content, "application/pdf")},
        data={
            "title": "Çift Anadal – Yandal Yönergesi",
            "source_url": "https://www.example.edu.tr/yonerge.pdf",
        },
    )


def only_document(client: TestClient) -> dict:
    body = client.get("/api/documents").json()
    assert body["total"] == 1
    return body["documents"][0]


def assert_rolled_back(
    client: TestClient, store: InMemoryVectorStore, documents_dir: Path
) -> dict:
    document = only_document(client)
    assert document["status"] == "failed"
    assert document["error_message"]  # Turkish text shown in the documents screen
    assert store.chunks_of(document["document_id"]) == []
    assert document["document_id"] in store.deleted
    leftovers = list(documents_dir.glob("*")) if documents_dir.exists() else []
    assert leftovers == [], leftovers  # neither the PDF nor a .part file
    return document


# --- controlled 4xx for bad PDFs ---------------------------------------------- #
def test_corrupt_pdf_returns_415_and_rolls_back(client, store, documents_dir) -> None:
    response = upload(client, pdf_factory.CORRUPT_PDF_BYTES)

    assert response.status_code == 415
    assert response.json()["error"] == "invalid_pdf"
    assert "PDF" in response.json()["message"]
    document = assert_rolled_back(client, store, documents_dir)
    assert document["error_message"] == response.json()["message"]


def test_password_protected_pdf_returns_422_and_rolls_back(
    client, store, documents_dir, tmp_path: Path
) -> None:
    response = upload(client, pdf_factory.encrypted_pdf_bytes(tmp_path))

    assert response.status_code == 422
    assert response.json()["error"] == "encrypted_pdf"
    assert "parola" in response.json()["message"]
    assert_rolled_back(client, store, documents_dir)


def test_scanned_pdf_without_text_returns_422_and_rolls_back(
    client, store, documents_dir, tmp_path: Path
) -> None:
    response = upload(client, pdf_factory.image_only_pdf_bytes(tmp_path))

    assert response.status_code == 422
    assert response.json()["error"] == "no_extractable_text"
    assert_rolled_back(client, store, documents_dir)


def test_pymupdf_crash_while_reading_is_422_not_500(
    client, store, documents_dir, tmp_path: Path, monkeypatch
) -> None:
    content = pdf_factory.build_text_pdf(tmp_path / "ok.pdf").read_bytes()

    def broken(*args, **kwargs):
        raise RuntimeError("code=2: zlib error: invalid stored block lengths")

    monkeypatch.setattr(pdf_extractor, "_page_blocks", broken)
    response = upload(client, content)

    assert response.status_code == 422
    assert response.json()["error"] == "unreadable_pdf"
    assert_rolled_back(client, store, documents_dir)


def test_not_a_pdf_is_rejected_before_anything_is_stored(
    client, store, documents_dir
) -> None:
    response = upload(client, b"Merhaba, ben bir metin dosyasiyim.", name="sahte.pdf")

    assert response.status_code == 415
    assert client.get("/api/documents").json()["total"] == 0
    assert not documents_dir.exists() or list(documents_dir.iterdir()) == []
    assert store.chunks == {}


# --- rollback when the vector store fails -------------------------------------- #
def test_qdrant_unavailable_rolls_back_file_and_partial_chunks(
    client, store, documents_dir, tmp_path: Path
) -> None:
    store.fail_upsert = VectorStoreUnavailableError("connection refused")
    store.write_before_failing = True  # some points were written before the failure
    content = pdf_factory.build_ruled_calendar(tmp_path / "takvim.pdf").read_bytes()

    response = upload(client, content)

    assert response.status_code == 503
    assert response.json()["error"] == "service_unavailable"
    assert_rolled_back(client, store, documents_dir)


def test_unexpected_vector_store_error_still_rolls_back(
    client, store, documents_dir, tmp_path: Path
) -> None:
    store.fail_upsert = RuntimeError("boom")
    content = pdf_factory.build_text_pdf(tmp_path / "ok.pdf").read_bytes()

    response = upload(client, content)

    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert_rolled_back(client, store, documents_dir)


# --- happy path and delete, for contrast ---------------------------------------- #
def test_successful_upload_then_delete_leaves_nothing_behind(
    client, store, documents_dir, tmp_path: Path
) -> None:
    content = pdf_factory.build_ruled_calendar(tmp_path / "takvim.pdf").read_bytes()

    response = upload(client, content)
    assert response.status_code == 201
    document = response.json()
    assert document["status"] == "indexed"
    document_id = document["document_id"]
    assert (documents_dir / f"{document_id}.pdf").exists()
    assert list(documents_dir.glob("*.part")) == []
    chunks = store.chunks_of(document_id)
    assert chunks and document["chunk_count"] == len(chunks)
    assert all("Güz Yarıyılı" in c.text for c in chunks if "Güz etkinliği" in c.text)

    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert list(documents_dir.iterdir()) == []
    assert store.chunks_of(document_id) == []
    assert client.get("/api/documents").json()["total"] == 0
