"""Page-aware text extraction from text-based PDFs using PyMuPDF.

Owner: Person 3 (document processing).

Every returned ``PageText`` carries the real 1-based page number that the user
sees in a PDF viewer, so source citations stay correct through the pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ApplicationError

logger = logging.getLogger(__name__)

# find_tables() would otherwise print an install hint to stdout on first use.
if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()

PDF_MAGIC = b"%PDF-"
_BOLD_FLAG = 1 << 4  # PyMuPDF span flag for bold fonts
_HEADING_SIZE_DELTA = 1.5  # pt larger than body text => heading candidate
_MAX_HEADING_CHARS = 120


def _max_file_bytes() -> int:
    return get_settings().max_upload_mb * 1024 * 1024


# --------------------------------------------------------------------------- #
# Errors: ``message`` is Turkish text for the UI (returned by the shared
# ApplicationError handler); ``detail`` is technical and stays in the logs.
# --------------------------------------------------------------------------- #
class PDFProcessingError(ApplicationError):
    """Base error for controlled PDF processing failures."""

    error_code = "pdf_processing_failed"
    default_message = "PDF dosyası işlenemedi."
    http_status = 422

    def __init__(self, detail: str = "", message: str | None = None) -> None:
        super().__init__(
            code=self.error_code,
            message=message or self.default_message,
            status_code=self.http_status,
        )
        self.detail = detail or self.message

    def __str__(self) -> str:
        return self.detail


class InvalidPDFError(PDFProcessingError):
    error_code = "invalid_pdf"
    default_message = "Dosya geçerli bir PDF değil veya bozuk."
    http_status = 415


class EncryptedPDFError(PDFProcessingError):
    error_code = "encrypted_pdf"
    default_message = "PDF parola korumalı. Lütfen parolasız bir kopya yükleyin."


class FileTooLargeError(PDFProcessingError):
    error_code = "file_too_large"
    default_message = "PDF dosyası izin verilen boyuttan büyük."
    http_status = 413


class NoExtractableTextError(PDFProcessingError):
    error_code = "no_extractable_text"
    default_message = (
        "PDF'de çıkarılabilir metin bulunamadı. Taranmış (görüntü tabanlı) "
        "belgeler desteklenmiyor."
    )


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PageText:
    """Raw text of one PDF page.

    ``text`` keeps one PDF line per ``\\n`` and separates PDF text blocks with a
    blank line, so the cleaner can rebuild paragraphs and detect repeated
    header/footer lines.
    ``headings`` are lines the PDF itself styles as headings (bold or larger
    font). They are only hints for ``section_title``; nothing is invented.
    """

    page_number: int
    text: str
    headings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["headings"] = list(self.headings)
        return data


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_pdf_bytes(content: bytes) -> None:
    """Cheap checks before a file is written to disk (used by upload)."""
    if not content:
        raise InvalidPDFError("Uploaded file is empty.")
    if len(content) > _max_file_bytes():
        raise FileTooLargeError(f"File size {len(content)} exceeds limit.")
    # The PDF spec allows junk before the header; readers accept it within 1 KB.
    if PDF_MAGIC not in content[:1024]:
        raise InvalidPDFError("Missing %PDF- header; extension alone is not trusted.")


def _open_pdf(file_path: Path) -> pymupdf.Document:
    path = Path(file_path)
    if not path.is_file():
        raise InvalidPDFError(f"File not found: {path}")
    size = path.stat().st_size
    if size > _max_file_bytes():
        raise FileTooLargeError(f"File size {size} exceeds limit: {path}")
    with path.open("rb") as handle:
        validate_pdf_bytes(handle.read(1024) or b"")

    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several internal types
        raise InvalidPDFError(f"PyMuPDF could not open {path}: {exc}") from exc

    if doc.needs_pass:
        doc.close()
        raise EncryptedPDFError(f"Password protected PDF: {path}")
    if doc.page_count == 0:
        doc.close()
        raise InvalidPDFError(f"PDF has no pages: {path}")
    return doc


def validate_pdf(file_path: Path) -> int:
    """Raise a ``PDFProcessingError`` unless the file opens as a usable PDF.

    Returns the page count.
    """
    doc = _open_pdf(file_path)
    try:
        return doc.page_count
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _line_text(line: dict[str, Any]) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _body_font_size(pages: list[dict[str, Any]]) -> float:
    """Most common font size weighted by character count across the document."""
    sizes: Counter[float] = Counter()
    for page in pages:
        for block in page["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _is_heading_line(line: dict[str, Any], text: str, body_size: float) -> bool:
    if not (3 <= len(text) <= _MAX_HEADING_CHARS):
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if text.endswith((",", ";")):
        return False
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return False
    all_bold = all(s["flags"] & _BOLD_FLAG for s in spans)
    larger = min(s["size"] for s in spans) >= body_size + _HEADING_SIZE_DELTA
    return all_bold or larger


# --------------------------------------------------------------------------- #
# Layout: lines, blocks and tables
# --------------------------------------------------------------------------- #
_TABLE_CELL_MAX_CHARS = 40  # longer lines are body text, never table cells
_TABLE_BLOCK_MAX_WIDTH = 0.45  # share of page width; protects two-column text
_TABLE_ROW_GAP_PT = 12.0  # blocks this close vertically belong to one table


@dataclass
class _Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    is_heading: bool = False

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class _Block:
    x0: float
    y0: float
    x1: float
    y1: float
    lines: list[_Line] = field(default_factory=list)
    table_text: str | None = None  # set for tables: rows as "a | b"

    @classmethod
    def from_lines(cls, lines: list[_Line]) -> _Block:
        return cls(
            min(ln.x0 for ln in lines),
            min(ln.y0 for ln in lines),
            max(ln.x1 for ln in lines),
            max(ln.y1 for ln in lines),
            lines,
        )

    @property
    def is_table(self) -> bool:
        return self.table_text is not None

    @property
    def text(self) -> str:
        return (
            self.table_text
            if self.table_text is not None
            else "\n".join(ln.text for ln in self.lines)
        )


def _cell(value: object) -> str:
    return " ".join(str(value or "").split())


def _ruled_tables(page: pymupdf.Page) -> list[_Block]:
    """Tables drawn with ruling lines (e.g. an academic calendar)."""
    try:
        found = page.find_tables()
    except Exception as exc:  # noqa: BLE001 - table detection is best effort
        logger.debug("Table detection failed on page %s: %s", page.number + 1, exc)
        return []
    tables = []
    for table in found.tables:
        if table.row_count < 2 or table.col_count < 2:
            continue
        rows = [" | ".join(_cell(c) for c in row) for row in table.extract()]
        rows = [row for row in rows if row.replace("|", "").strip()]
        if rows:
            x0, y0, x1, y1 = table.bbox
            tables.append(_Block(x0, y0, x1, y1, table_text="\n".join(rows)))
    return tables


def _without_table_lines(blocks: list[_Block], tables: list[_Block]) -> list[_Block]:
    """Drop lines inside table areas; split blocks around them."""

    def inside(line: _Line) -> bool:
        cx = (line.x0 + line.x1) / 2
        return any(t.x0 <= cx <= t.x1 and t.y0 <= line.y_center <= t.y1 for t in tables)

    result: list[_Block] = []
    for block in blocks:
        run: list[_Line] = []
        for line in block.lines:
            if inside(line):
                if run:
                    result.append(_Block.from_lines(run))
                run = []
            else:
                run.append(line)
        if run:
            result.append(_Block.from_lines(run))
    return result


def _rows_from_lines(lines: list[_Line]) -> str:
    """Group lines of side-by-side columns into rows: "AA | 4,00"."""
    ordered = sorted(lines, key=lambda ln: (ln.y_center, ln.x0))
    heights = sorted(ln.y1 - ln.y0 for ln in ordered)
    tolerance = heights[len(heights) // 2] * 0.5
    rows: list[list[_Line]] = []
    for line in ordered:
        if rows and line.y_center - rows[-1][0].y_center <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    return "\n".join(
        " | ".join(ln.text for ln in sorted(row, key=lambda ln: ln.x0)) for row in rows
    )


def _vertical_gap(a: _Block, b: _Block) -> float:
    return max(a.y0, b.y0) - min(a.y1, b.y1)  # negative when they overlap


def _side_by_side_tables(blocks: list[_Block], page_width: float) -> list[_Block]:
    """Turn borderless tables (columns printed as separate narrow blocks)
    into row-wise text. Normal full-width paragraphs are never touched."""
    candidates = [
        i
        for i, b in enumerate(blocks)
        if not b.is_table
        and (b.x1 - b.x0) <= _TABLE_BLOCK_MAX_WIDTH * page_width
        and all(len(ln.text) <= _TABLE_CELL_MAX_CHARS for ln in b.lines)
    ]
    parent = {i: i for i in candidates}

    def find(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for a_idx, i in enumerate(candidates):
        for j in candidates[a_idx + 1 :]:
            a, b = blocks[i], blocks[j]
            overlap = -_vertical_gap(a, b)
            horizontally_apart = a.x1 <= b.x0 or b.x1 <= a.x0
            if horizontally_apart and overlap >= 0.5 * min(a.y1 - a.y0, b.y1 - b.y0):
                parent[find(j)] = find(i)

    groups: dict[int, set[int]] = {}
    for i in candidates:
        groups.setdefault(find(i), set()).add(i)
    tables = [g for g in groups.values() if len(g) >= 2]
    if not tables:
        return blocks

    def span(group: set[int]) -> _Block:
        return _Block.from_lines([ln for i in group for ln in blocks[i].lines])

    # Attach nearby narrow blocks (a header line such as "Notu") and merge
    # table parts that follow each other closely.
    changed = True
    while changed:
        changed = False
        grouped = set().union(*tables)
        for i in candidates:
            if i in grouped:
                continue
            for group in tables:
                g = span(group)
                if (
                    _vertical_gap(g, blocks[i]) <= _TABLE_ROW_GAP_PT
                    and blocks[i].x0 < g.x1
                    and blocks[i].x1 > g.x0
                ):
                    group.add(i)
                    changed = True
                    break
        for a in range(len(tables)):
            for b in range(a + 1, len(tables)):
                ga, gb = span(tables[a]), span(tables[b])
                if (
                    _vertical_gap(ga, gb) <= _TABLE_ROW_GAP_PT
                    and ga.x0 < gb.x1
                    and gb.x0 < ga.x1
                ):
                    tables[a] |= tables.pop(b)
                    changed = True
                    break
            if changed:
                break

    used = set().union(*tables)
    result = [b for i, b in enumerate(blocks) if i not in used]
    for group in tables:
        lines = [ln for i in group for ln in blocks[i].lines]
        merged = _Block.from_lines(lines)
        merged.table_text = _rows_from_lines(lines)
        result.append(merged)
    return result


def _page_blocks(
    page: pymupdf.Page, raw: dict[str, Any], body_size: float
) -> list[_Block]:
    blocks: list[_Block] = []
    for block in raw["blocks"]:
        if block.get("type", 0) != 0:  # skip image blocks
            continue
        lines = []
        for line in block.get("lines", []):
            text = _line_text(line)
            if text:
                x0, y0, x1, y1 = line["bbox"]
                lines.append(
                    _Line(x0, y0, x1, y1, text, _is_heading_line(line, text, body_size))
                )
        if lines:
            blocks.append(_Block.from_lines(lines))

    tables = _ruled_tables(page)
    if tables:
        blocks = _without_table_lines(blocks, tables) + tables
    blocks = _side_by_side_tables(blocks, page.rect.width)
    return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))


def extract_pages(file_path: Path) -> list[PageText]:
    """Extract text per page, keeping real 1-based page numbers.

    Text blocks are separated by a blank line. Tables become one block whose
    rows are lines of ``cell | cell`` so that values stay next to their labels.
    Empty pages are returned with empty text (never dropped) so page numbering
    and empty-page statistics stay correct.
    """
    doc = _open_pdf(Path(file_path))
    try:
        raw_pages: list[dict[str, Any]] = []
        for page in doc:
            try:
                raw_pages.append(page.get_text("dict", sort=True))
            except Exception as exc:  # noqa: BLE001
                # One broken page must not fail the whole document.
                logger.warning("Page %s could not be read: %s", page.number + 1, exc)
                raw_pages.append({"blocks": []})

        doc_body_size = _body_font_size(raw_pages)
        result: list[PageText] = []
        for page, raw in zip(doc, raw_pages, strict=True):
            # Per-page body size copes with pages scaled differently.
            body_size = _body_font_size([raw]) or doc_body_size
            blocks = _page_blocks(page, raw, body_size)
            headings = [
                ln.text
                for b in blocks
                if not b.is_table
                for ln in b.lines
                if ln.is_heading
            ]
            result.append(
                PageText(
                    page_number=page.number + 1,
                    text="\n\n".join(b.text for b in blocks),
                    headings=tuple(headings),
                )
            )
    finally:
        doc.close()

    if all(page.is_empty for page in result):
        raise NoExtractableTextError(f"No text layer found in {file_path}")

    logger.info(
        "Extracted %s pages (%s empty, %s with tables) from %s",
        len(result),
        sum(p.is_empty for p in result),
        sum(" | " in p.text for p in result),
        Path(file_path).name,
    )
    return result
