"""Process one PDF and write pages.json + chunks.json (Person 3 deliverable).

Usage (from the repository root):
    python -m backend.scripts.process_pdf yonetmelik.pdf \
        --title "Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği" \
        --source-url "https://..." --document-id 3f6c...

Prints total pages, empty pages and chunk count, as required by the
acceptance criteria. Use the same --document-id to get identical chunk ids.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from backend.app.rag.pdf_extractor import PDFProcessingError
from backend.app.services.document_processor import PdfDocumentProcessor, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PDF -> pages.json + chunks.json")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--document-id", default=None, help="UUID; new one if omitted")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    document_id = args.document_id or str(uuid.uuid4())

    try:
        result = PdfDocumentProcessor().process(
            args.pdf,
            document_id=document_id,
            document_title=args.title,
            source_url=args.source_url,
        )
    except PDFProcessingError as exc:
        print(f"HATA: {exc.message}", file=sys.stderr)
        logging.getLogger(__name__).debug("detail: %s", exc.detail)
        return 1

    pages_path, chunks_path = write_outputs(result, args.out_dir / document_id)
    s = result.stats
    print("\n=== CampusLens belge işleme özeti ===")
    print(f"Belge ID          : {document_id}")
    print(f"Toplam sayfa      : {s['page_count']}")
    print(f"Boş sayfa         : {s['empty_page_count']} {s['empty_pages'] or ''}")
    print(f"Chunk sayısı      : {s['chunk_count']}")
    print(
        f"Token min/ort/max : {s['min_tokens']} / {s['avg_tokens']} / {s['max_tokens']}"
    )
    print(f"Başlıklı chunk    : {s['chunks_with_section_title']}/{s['chunk_count']}")
    print(f"Tokenizer         : {s['tokenizer']}")
    print(f"Süre              : {s['elapsed_ms']} ms")
    print(f"Çıktılar          : {pages_path}\n                    {chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
