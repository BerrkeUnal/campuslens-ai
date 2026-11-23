"""Index a chunks.json file (list of ChunkRecord objects) into Qdrant.

Usage (from the repository root):

    python scripts/index_documents.py --input backend/tests/fixtures/sample_chunks.json
    python scripts/index_documents.py --input chunks.json --recreate
    python scripts/index_documents.py --input chunks.json \
        --query "Bütünleme sınavına kimler girebilir?"

By default every document found in the file is replaced: the new chunks are
written first, then the document's points that are not in the file are
removed. QDRANT_URL decides where data goes: ``http://localhost:6333`` for a
Qdrant server, or a folder such as ``./data/qdrant_local`` for Qdrant's
embedded local mode (stop the backend first; a folder can be opened by only
one process at a time).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pydantic import TypeAdapter, ValidationError

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ApplicationError
from backend.app.rag.embedder import TextEmbedder
from backend.app.rag.qdrant_store import DEFAULT_UPSERT_BATCH_SIZE, QdrantStore
from backend.app.schemas.retrieval import ChunkRecord, RetrievedChunk

_CHUNK_LIST = TypeAdapter(list[ChunkRecord])


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
        description="Index ChunkRecord JSON into the CampusLens Qdrant collection.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to chunks.json produced by document processing.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the whole collection before indexing.",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Only upsert; keep points of the same documents that are not in the file.",
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

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for noisy_logger in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # PDF text may contain characters the Windows console code page lacks.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    settings = get_settings()

    try:
        chunks = load_chunks(args.input)
    except FileNotFoundError:
        print(f"Hata: dosya bulunamadı: {args.input}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Hata: dosya okunamadı: {args.input} ({exc})", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        print(
            f"Hata: {args.input} geçerli ChunkRecord listesi değil:\n{exc}",
            file=sys.stderr,
        )
        return 1

    if not chunks:
        print("Uyarı: dosyada chunk yok; hiçbir şey indekslenmedi.")
        return 0

    embedder = TextEmbedder(
        settings.embedding_model,
        settings.sparse_model,
        batch_size=args.batch_size,
    )

    try:
        store = QdrantStore(
            embedder=embedder,
            settings=settings,
            batch_size=args.batch_size,
        )
    except ApplicationError as exc:
        print(f"Hata: {exc.message}\nAyrıntı: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Hata: QDRANT_URL ayarı geçersiz.\nAyrıntı: {exc}", file=sys.stderr)
        return 2

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
        return 2
    except ValueError as exc:
        print(f"Hata: chunk verisi geçersiz.\nAyrıntı: {exc}", file=sys.stderr)
        return 1
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
