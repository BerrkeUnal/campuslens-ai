"""Index documents into the CampusLens Qdrant collection from the command line.

Usage (from the repository root):

    # One PDF, exactly like the admin upload: SQLite row + data/documents + Qdrant,
    # so it is listed in the admin screen and can be deleted there.
    python scripts/index_documents.py --pdf yonetmelik.pdf \
        --title "Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği" \
        --source-url "https://www.istinye.edu.tr/...pdf"

    # Every PDF of the catalog that exists in a folder (titles and source URLs
    # come from the catalog, so they match the evaluation gold data).
    python scripts/index_documents.py --catalog resources/istinye_documents.json \
        --pdf-dir path/to/pdfs

    # Low-level, for tests and debugging: a chunks.json file straight into
    # Qdrant. These chunks are NOT added to the app's document list.
    python scripts/index_documents.py --input backend/tests/fixtures/sample_chunks.json
    python scripts/index_documents.py --input chunks.json --recreate

Every mode accepts ``--query "..."`` to print dense, BM25 and hybrid results
afterwards. ``--pdf``/``--catalog`` skip a PDF whose source URL is already
indexed; ``--replace`` indexes it again and then removes the old copy.
With ``--input`` every document in the file is replaced: the new chunks are
written first, then the document's points that are not in the file are removed.

QDRANT_URL decides where data goes: ``http://localhost:6333`` for a Qdrant
server, or a folder such as ``./data/qdrant_local`` for Qdrant's embedded local
mode (stop the backend first; a folder can be opened by only one process at a
time). ``--pdf``/``--catalog`` also use DATABASE_URL, like the backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pydantic import TypeAdapter, ValidationError

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import ApplicationError
from backend.app.db.database import SQLiteDatabase
from backend.app.rag.embedder import TextEmbedder
from backend.app.rag.qdrant_store import (
    DEFAULT_UPSERT_BATCH_SIZE,
    QdrantStore,
    QdrantVectorStore,
    VectorStoreUnavailableError,
)
from backend.app.schemas.documents import (
    DocumentCreate,
    DocumentResponse,
    DocumentStatus,
)
from backend.app.schemas.retrieval import ChunkRecord, RetrievedChunk
from backend.app.services.document_ingestion_service import DocumentIngestionService
from backend.app.services.document_processor import (
    DOCUMENTS_DIR,
    get_document_processor,
)
from backend.app.services.document_service import (
    DocumentService,
    SQLiteDocumentService,
)

_CHUNK_LIST = TypeAdapter(list[ChunkRecord])

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_SERVICE_ERROR = 2


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Read ChunkRecord objects from a JSON list or a {"chunks": [...]} object."""
    # utf-8-sig also accepts files saved with a BOM (e.g. Windows Notepad).
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, dict) and "chunks" in data:
        data = data["chunks"]

    if not isinstance(data, list):
        raise TypeError(
            "Input must be a JSON list of chunks or an object with a 'chunks' list.",
        )

    return _CHUNK_LIST.validate_python(data)


# PDF ingestion (same path as POST /api/documents) ----------------------------------


@dataclass(frozen=True)
class PdfSource:
    path: Path
    title: str
    source_url: str


@dataclass(frozen=True)
class PdfOutcome:
    source: PdfSource
    status: str  # "indexed" | "skipped" | "failed" | "missing"
    detail: str = ""
    document: DocumentResponse | None = None


def load_catalog(catalog_path: Path, pdf_dir: Path) -> list[PdfSource]:
    """Read resources/istinye_documents.json-style entries as PDF sources."""
    with catalog_path.open(encoding="utf-8-sig") as file:
        data = json.load(file)

    entries = data.get("documents") if isinstance(data, dict) else data

    if not isinstance(entries, list):
        raise TypeError("Catalog must be a list or an object with a 'documents' list.")

    sources = []

    for index, entry in enumerate(entries, start=1):
        fields = {
            key: entry.get(key) if isinstance(entry, dict) else None
            for key in ("title", "file_name", "source_url")
        }

        if not all(
            isinstance(value, str) and value.strip() for value in fields.values()
        ):
            raise TypeError(
                f"Catalog entry {index} needs non-empty 'title', 'file_name' "
                "and 'source_url' strings.",
            )

        sources.append(
            PdfSource(
                # Only the file name is used, so an entry cannot point outside pdf_dir.
                path=pdf_dir / Path(fields["file_name"]).name,
                title=fields["title"],
                source_url=fields["source_url"],
            ),
        )

    return sources


async def ingest_pdfs(
    sources: list[PdfSource],
    *,
    ingestion: DocumentIngestionService,
    documents: DocumentService,
    replace: bool = False,
) -> list[PdfOutcome]:
    """Ingest PDFs one by one; a failing PDF does not stop the others."""
    outcomes = []

    for source in sources:
        outcomes.append(
            await _ingest_pdf(
                source,
                ingestion=ingestion,
                documents=documents,
                replace=replace,
            ),
        )

    return outcomes


async def _ingest_pdf(
    source: PdfSource,
    *,
    ingestion: DocumentIngestionService,
    documents: DocumentService,
    replace: bool,
) -> PdfOutcome:
    if not source.path.is_file():
        return PdfOutcome(source, "missing", f"dosya bulunamadı: {source.path}")

    try:
        metadata = DocumentCreate(title=source.title, source_url=source.source_url)
    except ValidationError as exc:
        return PdfOutcome(source, "failed", f"başlık veya kaynak URL geçersiz: {exc}")

    try:
        listed = (await documents.list_documents()).documents
    except ApplicationError as exc:
        return PdfOutcome(source, "failed", f"belge listesi okunamadı: {exc.message}")

    previous = [doc for doc in listed if doc.source_url == metadata.source_url]
    indexed = [doc for doc in previous if doc.status == DocumentStatus.INDEXED]

    if indexed and not replace:
        return PdfOutcome(
            source,
            "skipped",
            f"bu kaynak URL zaten indekslenmiş ({indexed[0].document_id}); "
            "yeniden yüklemek için --replace kullanın",
            indexed[0],
        )

    try:
        content = await asyncio.to_thread(source.path.read_bytes)
    except OSError as exc:
        return PdfOutcome(source, "failed", f"dosya okunamadı: {exc}")

    try:
        document = await ingestion.ingest_document(metadata, source.path.name, content)
    except ApplicationError as exc:
        detail = exc.message if str(exc) == exc.message else f"{exc.message} ({exc})"
        return PdfOutcome(source, "failed", detail)
    # One unreadable PDF must not stop the rest; the service already logged it.
    except Exception as exc:  # noqa: BLE001
        return PdfOutcome(source, "failed", f"{type(exc).__name__}: {exc}")

    if replace:
        # The new copy is indexed first; only then the old copies are removed.
        for old in previous:
            try:
                await ingestion.delete_document(old.document_id)
            except Exception as exc:  # noqa: BLE001 - reported in the summary
                return PdfOutcome(
                    source,
                    "failed",
                    f"yeni kopya indekslendi ({document.document_id}) ama eski kopya "
                    f"{old.document_id} silinemedi: {exc}",
                    document,
                )

    return PdfOutcome(source, "indexed", "", document)


def format_outcome(outcome: PdfOutcome) -> str:
    title = outcome.source.title
    document = outcome.document

    if outcome.status == "indexed" and document is not None and not outcome.detail:
        return (
            f"  [OK]      {title}: {document.page_count} sayfa, "
            f"{document.chunk_count} parça ({document.document_id})"
        )

    labels = {"skipped": "[ATLANDI]", "failed": "[HATA]", "missing": "[YOK]"}
    return (
        f"  {labels.get(outcome.status, outcome.status):<9} {title}: {outcome.detail}"
    )


# Output and arguments ------------------------------------------------------------


def format_results(title: str, results: list[RetrievedChunk], top: int) -> str:
    lines = [f"  {title}:"]

    if not results:
        lines.append("    (sonuç yok)")

    for rank, chunk in enumerate(results[:top], start=1):
        excerpt = " ".join(chunk.text.split())[:90]
        lines.append(
            f"    {rank}. {chunk.retrieval_score:.4f}  "
            f"{chunk.document_title} s.{chunk.page_number}  "
            f"[{chunk.chunk_id}]\n       {excerpt}...",
        )

    return "\n".join(lines)


def print_search_results(store: QdrantStore, queries: list[str], top: int) -> None:
    for question in queries:
        print(f"\nSoru: {question}")
        print(format_results("dense", store.search_dense(question, top), top))
        print(format_results("bm25", store.search_bm25(question, top), top))
        print(format_results("hybrid", store.search_hybrid_rrf(question, top), top))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index documents into the CampusLens Qdrant collection.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pdf",
        type=Path,
        help="PDF to ingest like the admin upload (needs --title and --source-url).",
    )
    source.add_argument(
        "--catalog",
        type=Path,
        help="Catalog JSON (e.g. resources/istinye_documents.json); needs --pdf-dir.",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="chunks.json to write straight into Qdrant (not listed in the app).",
    )
    parser.add_argument("--title", help="Document title for --pdf.")
    parser.add_argument("--source-url", help="Official source URL for --pdf.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        help="Folder that contains the catalog's PDF files (--catalog).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="--pdf/--catalog: index again even if the source URL is already indexed.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="--input only: drop and recreate the whole collection before indexing.",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="--input only: keep points of the same documents that are not in the file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_UPSERT_BATCH_SIZE,
        help="Chunks per embedding/upsert batch (default: 64).",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Question to search after indexing; may be repeated.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of results to print per method (default: 5).",
    )
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size en az 1 olmalı")

    if args.top < 1:
        parser.error("--top en az 1 olmalı")

    if any(not question.strip() for question in args.query):
        parser.error("--query boş olamaz")

    if args.pdf is not None:
        if not (args.title and args.title.strip()):
            parser.error("--pdf için --title gerekli")
        if not (args.source_url and args.source_url.strip()):
            parser.error("--pdf için --source-url gerekli")
    elif args.title is not None or args.source_url is not None:
        parser.error("--title ve --source-url yalnızca --pdf ile kullanılır")

    if args.catalog is not None and args.pdf_dir is None:
        parser.error("--catalog için --pdf-dir gerekli")
    if args.catalog is None and args.pdf_dir is not None:
        parser.error("--pdf-dir yalnızca --catalog ile kullanılır")

    if args.input is None and (args.recreate or args.no_replace):
        parser.error("--recreate ve --no-replace yalnızca --input ile kullanılır")
    if args.input is not None and args.replace:
        parser.error("--replace yalnızca --pdf veya --catalog ile kullanılır")

    return args


# Entry point ---------------------------------------------------------------------


def main(argv: list[str] | None = None, *, root: Path = REPOSITORY_ROOT) -> int:
    args = parse_args(argv)

    # File arguments are relative to where the command was typed. Settings
    # (.env, DATABASE_URL, a QDRANT_URL folder, data/documents) are relative
    # to the repository root, exactly as for the backend (scripts/run_dev.py).
    for name in ("pdf", "catalog", "pdf_dir", "input"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    os.chdir(root)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for noisy_logger in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # PDF text may contain characters the Windows console code page lacks.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    settings = get_settings()

    if args.input is not None:
        return _index_chunks_file(args, settings)

    return _ingest_pdf_files(args, settings)


def _open_store(args: argparse.Namespace, settings: Settings) -> QdrantStore | None:
    embedder = TextEmbedder(
        settings.embedding_model,
        settings.sparse_model,
        batch_size=args.batch_size,
    )

    try:
        return QdrantStore(
            embedder=embedder,
            settings=settings,
            batch_size=args.batch_size,
        )
    except ValueError as exc:  # includes VectorStoreConfigError
        print(f"Hata: QDRANT_URL ayarı geçersiz.\nAyrıntı: {exc}", file=sys.stderr)
    except VectorStoreUnavailableError as exc:
        print(f"Hata: {exc.message}\nAyrıntı: {exc}", file=sys.stderr)
        if not settings.qdrant_url.strip().lower().startswith(("http://", "https://")):
            print(
                "İpucu: yerel Qdrant klasörünü aynı anda tek süreç açabilir; "
                "önce backend'i durdurun.",
                file=sys.stderr,
            )
    except ApplicationError as exc:
        print(f"Hata: {exc.message}\nAyrıntı: {exc}", file=sys.stderr)

    return None


def _ingest_pdf_files(args: argparse.Namespace, settings: Settings) -> int:
    if args.pdf is not None:
        sources = [PdfSource(args.pdf, args.title.strip(), args.source_url.strip())]
    else:
        try:
            sources = load_catalog(args.catalog, args.pdf_dir)
        except FileNotFoundError:
            print(f"Hata: katalog bulunamadı: {args.catalog}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"Hata: katalog okunamadı: {args.catalog}\n{exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR

    try:
        database = SQLiteDatabase(settings.database_url)
    except ValueError as exc:
        print(f"Hata: DATABASE_URL ayarı geçersiz: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    store = _open_store(args, settings)

    if store is None:
        return EXIT_SERVICE_ERROR

    started = time.perf_counter()

    try:
        try:
            # Fail before any PDF is stored or gets a SQLite row.
            store.create_collection()
        except ApplicationError as exc:
            print(
                f"Hata: Qdrant hazır değil, hiçbir belge işlenmedi.\n{exc.message}\n"
                f"Ayrıntı: {exc}",
                file=sys.stderr,
            )
            return EXIT_SERVICE_ERROR

        try:
            outcomes = asyncio.run(
                _run_ingestion(database, store, sources, args.replace),
            )
        except ApplicationError as exc:  # e.g. the SQLite file cannot be opened
            print(f"Hata: {exc.message}\nAyrıntı: {exc}", file=sys.stderr)
            return EXIT_SERVICE_ERROR

        counts = Counter(outcome.status for outcome in outcomes)

        print("\nBelge yükleme özeti")
        print(f"  Veritabanı : {database.path}")
        print(f"  PDF klasörü: {DOCUMENTS_DIR.resolve()}")
        print(f"  Qdrant     : {settings.qdrant_url} ({store.collection_name})")
        for outcome in outcomes:
            print(format_outcome(outcome))
        print(
            f"  Toplam     : {counts['indexed']} indekslendi, "
            f"{counts['skipped']} atlandı, {counts['failed']} hata, "
            f"{counts['missing']} bulunamadı "
            f"({time.perf_counter() - started:.1f} sn)",
        )

        if counts["failed"]:
            exit_code = EXIT_SERVICE_ERROR
        else:
            exit_code = EXIT_INPUT_ERROR if counts["missing"] else EXIT_OK

        if args.query:
            try:
                print_search_results(store, args.query, args.top)
            except ApplicationError as exc:
                print(f"Hata: {exc.message}\nAyrıntı: {exc}", file=sys.stderr)
                exit_code = EXIT_SERVICE_ERROR

        return exit_code
    finally:
        store.close()


async def _run_ingestion(
    database: SQLiteDatabase,
    store: QdrantStore,
    sources: list[PdfSource],
    replace: bool,
) -> list[PdfOutcome]:
    await database.initialize()
    documents = SQLiteDocumentService(database)
    ingestion = DocumentIngestionService(
        documents,
        get_document_processor(),
        QdrantVectorStore(store),
    )
    return await ingest_pdfs(
        sources,
        ingestion=ingestion,
        documents=documents,
        replace=replace,
    )


def _index_chunks_file(args: argparse.Namespace, settings: Settings) -> int:
    try:
        chunks = load_chunks(args.input)
    except FileNotFoundError:
        print(f"Hata: dosya bulunamadı: {args.input}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except OSError as exc:
        print(f"Hata: dosya okunamadı: {args.input} ({exc})", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        print(
            f"Hata: {args.input} geçerli ChunkRecord listesi değil:\n{exc}",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR

    if not chunks:
        print("Uyarı: dosyada chunk yok; hiçbir şey indekslenmedi.")
        return EXIT_OK

    store = _open_store(args, settings)

    if store is None:
        return EXIT_SERVICE_ERROR

    started = time.perf_counter()

    try:
        store.create_collection(recreate=args.recreate)

        if args.no_replace:
            store.upsert_chunks(chunks)
        else:
            store.replace_document_chunks(chunks)

        total_points = store.count_points()

        if args.query:
            print_search_results(store, args.query, args.top)
    except ApplicationError as exc:
        print(f"Hata: {exc.message}\nAyrıntı: {exc}", file=sys.stderr)
        return EXIT_SERVICE_ERROR
    except ValueError as exc:
        print(f"Hata: chunk verisi geçersiz.\nAyrıntı: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    finally:
        store.close()

    elapsed = time.perf_counter() - started
    documents = Counter(chunk.document_id for chunk in chunks)
    unique_chunks = len({chunk.chunk_id for chunk in chunks})

    print("\nİndeksleme özeti")
    print(f"  Qdrant           : {settings.qdrant_url}")
    print(f"  Collection       : {store.collection_name}")
    print(f"  Belge sayısı     : {len(documents)}")
    print(f"  Okunan chunk     : {len(chunks)}")
    print(f"  Yazılan chunk    : {unique_chunks}")
    print(f"  Collection toplam: {total_points} point")
    print(f"  Süre             : {elapsed:.1f} sn")
    print(
        "\nNot: --input ile yazılan belgeler uygulamanın belge listesine eklenmez "
        "(admin ekranında görünmez, oradan silinemez). Uygulamada görünmesi için "
        "--pdf veya --catalog kullanın.",
    )

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
