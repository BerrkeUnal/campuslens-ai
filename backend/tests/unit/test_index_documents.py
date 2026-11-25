"""Offline tests for scripts/index_documents.py (fake embedder and processor, no models)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest
from qdrant_client import QdrantClient, models

from backend.app.core.config import Settings
from backend.app.db.database import SQLiteDatabase
from backend.app.rag.qdrant_store import (
    QdrantStore,
    QdrantVectorStore,
    VectorStoreUnavailableError,
)
from backend.app.schemas.documents import DocumentStatus
from backend.app.schemas.retrieval import ChunkRecord
from backend.app.services.document_ingestion_service import DocumentIngestionService
from backend.app.services.document_service import SQLiteDocumentService

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CATALOG = REPOSITORY_ROOT / "resources" / "istinye_documents.json"
URL = "https://www.example.edu.tr/sinav-yonetmeligi.pdf"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "index_documents_script",
        REPOSITORY_ROOT / "scripts" / "index_documents.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


index_documents = _load_script()
PdfSource = index_documents.PdfSource


class FakeEmbedder:
    """Hash-based vectors; same interface as TextEmbedder."""

    dense_model_name = "fake-dense"
    sparse_model_name = "fake-sparse"
    dense_dimension = 8

    def __init__(self, *args, **kwargs) -> None:
        pass

    def load(self) -> None:
        pass

    def _vector(self, text: str) -> list[float]:
        rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
        vector = rng.normal(size=self.dense_dimension)
        return (vector / np.linalg.norm(vector)).tolist()

    def encode_documents(self, texts):
        return [self._vector(text) for text in texts]

    def encode_query(self, question):
        return self._vector(question)

    def _sparse(self, text: str) -> models.SparseVector:
        terms = sorted({zlib.crc32(word.encode()) for word in text.lower().split()})
        return models.SparseVector(indices=terms, values=[1.0] * len(terms))

    def encode_sparse(self, texts):
        return [self._sparse(text) for text in texts]

    def encode_sparse_query(self, question):
        return self._sparse(question)


class FakeProcessor:
    """Two chunks per PDF, built from the SQLite document like the real one."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def process_document(self, file_path: Path, document) -> list[ChunkRecord]:
        self.calls += 1
        assert file_path.is_file()

        if self.error is not None:
            raise self.error

        return [
            ChunkRecord(
                chunk_id=f"{document.document_id}-p{page}-c1",
                document_id=str(document.document_id),
                document_title=document.title,
                source_url=document.source_url,
                page_number=page,
                text=f"{document.title} sayfa {page} bütünleme sınavı",
            )
            for page in (1, 2)
        ]


def make_pdf(path: Path, text: str = "MADDE 1 - Bütünleme sınavı") -> Path:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    document.save(path)
    document.close()
    return path


@pytest.fixture
def env(tmp_path: Path):
    database = SQLiteDatabase(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    asyncio.run(database.initialize())
    documents = SQLiteDocumentService(database)
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, embedder=FakeEmbedder(), collection_name="cli")
    processor = FakeProcessor()
    ingestion = DocumentIngestionService(
        documents,
        processor,
        QdrantVectorStore(store),
        documents_dir=tmp_path / "documents",
    )
    yield SimpleNamespace(
        documents=documents,
        store=store,
        processor=processor,
        ingestion=ingestion,
        documents_dir=tmp_path / "documents",
        tmp=tmp_path,
    )
    client.close()


def ingest(env, sources, *, replace: bool = False):
    return asyncio.run(
        index_documents.ingest_pdfs(
            sources,
            ingestion=env.ingestion,
            documents=env.documents,
            replace=replace,
        ),
    )


def listed(env):
    return asyncio.run(env.documents.list_documents()).documents


def test_pdf_is_listed_and_deletable_like_an_admin_upload(env) -> None:
    pdf = make_pdf(env.tmp / "sinav.pdf")

    outcomes = ingest(env, [PdfSource(pdf, "Sınav Yönetmeliği", URL)])

    assert [outcome.status for outcome in outcomes] == ["indexed"]
    [document] = listed(env)
    assert document.status == DocumentStatus.INDEXED
    assert (document.title, document.source_url) == ("Sınav Yönetmeliği", URL)
    assert (document.page_count, document.chunk_count) == (2, 2)
    assert document.file_name == "sinav.pdf"
    assert (env.documents_dir / f"{document.document_id}.pdf").is_file()
    assert env.store.count_points(str(document.document_id)) == 2

    # The same delete the admin screen uses removes every trace.
    assert asyncio.run(env.ingestion.delete_document(document.document_id))
    assert listed(env) == []
    assert env.store.count_points(str(document.document_id)) == 0
    assert not (env.documents_dir / f"{document.document_id}.pdf").exists()


def test_indexed_source_url_is_skipped_unless_replace(env) -> None:
    pdf = make_pdf(env.tmp / "sinav.pdf")
    source = PdfSource(pdf, "Sınav Yönetmeliği", URL)

    [first] = ingest(env, [source])
    [second] = ingest(env, [source])

    assert second.status == "skipped"
    assert "--replace" in second.detail
    assert env.processor.calls == 1
    assert len(listed(env)) == 1

    [third] = ingest(env, [source], replace=True)

    assert third.status == "indexed"
    [document] = listed(env)
    assert document.document_id == third.document.document_id
    assert document.document_id != first.document.document_id
    assert env.store.count_points(str(first.document.document_id)) == 0
    assert env.store.count_points() == 2  # no duplicate chunks in search


def test_one_bad_pdf_does_not_stop_the_others(env) -> None:
    good = make_pdf(env.tmp / "good.pdf")
    not_pdf = env.tmp / "not.pdf"
    not_pdf.write_bytes(b"%PDF-1.7 this is not really a PDF")

    outcomes = ingest(
        env,
        [
            PdfSource(env.tmp / "missing.pdf", "Eksik Belge", URL + "?missing"),
            PdfSource(good, "x", URL + "?short-title"),
            PdfSource(not_pdf, "Bozuk Belge", URL + "?broken"),
            PdfSource(good, "Sağlam Belge", URL),
        ],
    )

    assert [outcome.status for outcome in outcomes] == [
        "missing",
        "failed",
        "failed",
        "indexed",
    ]
    assert "missing.pdf" in outcomes[0].detail
    assert "geçersiz" in outcomes[1].detail
    statuses = {document.title: document.status for document in listed(env)}
    assert statuses == {
        "Bozuk Belge": DocumentStatus.FAILED,
        "Sağlam Belge": DocumentStatus.INDEXED,
    }
    assert env.store.count_points() == 2


def test_processing_crash_is_reported_and_marked_failed(env) -> None:
    env.processor.error = RuntimeError("code=7: Invalid number of pages")
    pdf = make_pdf(env.tmp / "sinav.pdf")

    [outcome] = ingest(env, [PdfSource(pdf, "Sınav Yönetmeliği", URL)])

    assert outcome.status == "failed"
    assert "Invalid number of pages" in outcome.detail
    [document] = listed(env)
    assert document.status == DocumentStatus.FAILED
    assert env.store.count_points() == 0
    assert "[HATA]" in index_documents.format_outcome(outcome)


def test_catalog_entries_keep_official_titles_and_urls(tmp_path: Path) -> None:
    sources = index_documents.load_catalog(CATALOG, tmp_path)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["documents"]

    assert len(sources) == len(catalog) == 9
    for source, entry in zip(sources, catalog, strict=True):
        assert source.path == tmp_path / entry["file_name"]
        assert (source.title, source.source_url) == (
            entry["title"],
            entry["source_url"],
        )


def test_catalog_paths_stay_inside_the_pdf_folder(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "documents": [
                    {"title": "T", "file_name": "../../x.pdf", "source_url": URL}
                ]
            },
        ),
        encoding="utf-8",
    )

    [source] = index_documents.load_catalog(catalog, tmp_path / "pdfs")

    assert source.path == tmp_path / "pdfs" / "x.pdf"


@pytest.mark.parametrize(
    "documents",
    [{"title": "T"}, [{"title": "T", "file_name": "", "source_url": URL}], "text"],
)
def test_malformed_catalog_is_rejected(tmp_path: Path, documents) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"documents": documents}), encoding="utf-8")

    with pytest.raises(TypeError):
        index_documents.load_catalog(catalog, tmp_path)


@pytest.mark.parametrize(
    "argv",
    [
        ["--pdf", "a.pdf", "--source-url", URL],
        ["--pdf", "a.pdf", "--title", "T"],
        ["--pdf", "a.pdf", "--title", "  ", "--source-url", URL],
        ["--pdf", "a.pdf", "--title", "T", "--source-url", URL, "--recreate"],
        ["--catalog", "c.json"],
        ["--input", "c.json", "--pdf-dir", "pdfs"],
        ["--input", "c.json", "--replace"],
        ["--input", "c.json", "--title", "T"],
        ["--input", "c.json", "--pdf", "a.pdf"],
        [],
    ],
)
def test_invalid_argument_combinations_are_rejected(argv) -> None:
    with pytest.raises(SystemExit) as caught:
        index_documents.parse_args(argv)

    assert caught.value.code == 2


def test_valid_argument_combinations() -> None:
    pdf = index_documents.parse_args(
        ["--pdf", "a.pdf", "--title", "T", "--source-url", URL, "--replace"],
    )
    catalog = index_documents.parse_args(
        ["--catalog", "c.json", "--pdf-dir", "pdfs", "--query", "staj"],
    )
    chunks = index_documents.parse_args(["--input", "c.json", "--recreate"])

    assert (pdf.pdf, pdf.replace) == (Path("a.pdf"), True)
    assert (catalog.catalog, catalog.pdf_dir) == (Path("c.json"), Path("pdfs"))
    assert chunks.recreate is True


# main(): runs from the repository root like the backend ------------------------------


@pytest.fixture
def cli(tmp_path: Path, monkeypatch):
    """A fake repository root with relative settings, and a different cwd."""
    root = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # main() changes cwd; monkeypatch restores it

    def configure(**overrides) -> Settings:
        values = {"qdrant_url": ":memory:", "database_url": "sqlite:///./data/cli.db"}
        settings = Settings(**(values | overrides))
        monkeypatch.setattr(index_documents, "get_settings", lambda: settings)
        return settings

    configure()
    monkeypatch.setattr(index_documents, "TextEmbedder", FakeEmbedder)
    monkeypatch.setattr(index_documents, "get_document_processor", FakeProcessor)
    return SimpleNamespace(root=root, elsewhere=elsewhere, configure=configure)


def run_main(cli, argv: list[str]) -> int:
    return index_documents.main(argv, root=cli.root)


def rows(database_path: Path) -> list:
    database = SQLiteDatabase(f"sqlite:///{database_path.as_posix()}")
    return asyncio.run(SQLiteDocumentService(database).list_documents()).documents


def test_main_pdf_mode_uses_repository_paths_from_any_folder(cli, capsys) -> None:
    make_pdf(cli.elsewhere / "sinav.pdf")

    # A relative --pdf is read from the current folder ...
    exit_code = run_main(
        cli,
        ["--pdf", "sinav.pdf", "--title", "Sınav Yönetmeliği", "--source-url", URL],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[OK]" in output
    assert "1 indekslendi" in output
    # ... but DATABASE_URL and data/documents are relative to the repository root.
    [document] = rows(cli.root / "data" / "cli.db")
    assert document.status == DocumentStatus.INDEXED
    assert (cli.root / "data" / "documents" / f"{document.document_id}.pdf").is_file()
    assert not (cli.elsewhere / "data").exists()


def test_main_catalog_mode_reports_missing_pdfs(cli, capsys) -> None:
    (cli.elsewhere / "pdfs").mkdir()

    exit_code = run_main(cli, ["--catalog", str(CATALOG), "--pdf-dir", "pdfs"])

    assert exit_code == 1
    assert capsys.readouterr().out.count("[YOK]") == 9


def test_main_stops_before_storing_anything_when_qdrant_is_down(
    cli,
    monkeypatch,
    capsys,
) -> None:
    def qdrant_down(self, recreate: bool = False) -> None:
        raise VectorStoreUnavailableError("connection refused")

    monkeypatch.setattr(QdrantStore, "create_collection", qdrant_down)
    make_pdf(cli.elsewhere / "sinav.pdf")

    exit_code = run_main(
        cli,
        ["--pdf", "sinav.pdf", "--title", "Sınav Yönetmeliği", "--source-url", URL],
    )

    assert exit_code == 2
    assert "hiçbir belge işlenmedi" in capsys.readouterr().err
    assert not (cli.root / "data").exists()  # no SQLite row, no stored PDF


def test_main_prints_the_summary_even_if_the_search_fails(
    cli,
    monkeypatch,
    capsys,
) -> None:
    def search_fails(self, question: str, limit: int = 20):
        raise VectorStoreUnavailableError("timeout")

    monkeypatch.setattr(QdrantStore, "search_dense", search_fails)
    make_pdf(cli.elsewhere / "sinav.pdf")

    exit_code = run_main(
        cli,
        [
            "--pdf",
            "sinav.pdf",
            "--title",
            "Sınav Yönetmeliği",
            "--source-url",
            URL,
            "--query",
            "staj",
        ],
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "1 indekslendi" in captured.out
    assert "Qdrant" in captured.err


def test_main_reports_unusable_qdrant_url(cli, capsys) -> None:
    cli.configure(qdrant_url="localhost:6333")
    chunks = REPOSITORY_ROOT / "backend" / "tests" / "fixtures" / "sample_chunks.json"

    exit_code = run_main(cli, ["--input", str(chunks)])

    assert exit_code == 2
    assert "QDRANT_URL ayarı geçersiz" in capsys.readouterr().err
