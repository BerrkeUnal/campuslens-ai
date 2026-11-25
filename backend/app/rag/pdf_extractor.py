"""Page-aware text extraction from text-based PDFs using PyMuPDF.

Owner: Person 3 (document processing).

Every returned ``PageText`` carries the real 1-based page number that the user
sees in a PDF viewer, so source citations stay correct through the pipeline.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import pairwise
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


class UnreadablePDFError(PDFProcessingError):
    error_code = "unreadable_pdf"
    default_message = (
        "PDF dosyası açıldı ancak içeriği okunamadı. Dosya bozuk olabilir; "
        "lütfen PDF'i yeniden kaydedip tekrar yükleyin."
    )


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
    # Tables keep their cells; text is rendered later with header/section context.
    table_rows: list[list[str]] | None = None
    rendered: str | None = None

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
        return self.table_rows is not None

    @property
    def text(self) -> str:
        if self.rendered is not None:
            return self.rendered
        return "\n".join(ln.text for ln in self.lines)


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
        rows = [[_cell(c) for c in row] for row in table.extract()]
        rows = [row for row in rows if any(row)]
        if rows:
            x0, y0, x1, y1 = table.bbox
            tables.append(_Block(x0, y0, x1, y1, table_rows=rows))
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


# --------------------------------------------------------------------------- #
# Table rendering: every row carries its column headers and section label
# --------------------------------------------------------------------------- #
_DIGIT = re.compile(r"\d")
# Section rows of academic calendars: "GÜZ YARIYILI", "Bahar Dönemi", "Yaz Okulu".
_SECTION_WORDS = re.compile(
    r"yar[ıi]y[ıi]l|d[öo]nem|yaz okulu|g[üu]z|bahar|semester|term", re.IGNORECASE
)


def _has_digit(cells: list[str]) -> bool:
    return any(_DIGIT.search(c) for c in cells)


def _section_label(row: list[str]) -> str | None:
    """A row whose only content is one label (a merged cell), e.g. "Güz Yarıyılı"."""
    values = {c for c in row if c}
    if len(row) >= 2 and len(values) == 1:
        return values.pop()
    return None


@dataclass
class _TableContext:
    """What a table continued on the next page still needs to know."""

    columns: int
    headers: list[str] | None
    section: str | None


def _render_table(
    rows: list[list[str]], previous: _TableContext | None
) -> tuple[list[str], list[str], _TableContext]:
    """Turn table cells into self-contained lines.

    ``Güz Yarıyılı – Etkinlik: Ders kayıtları | Başlangıç: 14.09.2026``

    Column headers are written into every row and the current section label
    is prefixed, so a row keeps its meaning even when the table is split
    across chunks or pages. Returns (lines, section labels, context).
    """
    columns = max(len(r) for r in rows)
    rows = [r + [""] * (columns - len(r)) for r in rows]

    headers: list[str] | None = None
    header_index: int | None = None
    body = [i for i, r in enumerate(rows) if _section_label(r) is None]
    if body:
        first = rows[body[0]]
        rest = [rows[i] for i in body[1:]]
        # Header row: several labels without numbers, above rows that have numbers.
        if (
            sum(bool(c) for c in first) >= 2
            and not _has_digit(first)
            and any(_has_digit(r) for r in rest)
        ):
            headers, header_index = first, body[0]

    section: str | None = None
    if previous is not None and previous.columns == columns:
        section = previous.section
        if headers is None:
            headers = previous.headers  # table continues from the previous page

    lines: list[str] = []
    sections: list[str] = []
    for index, row in enumerate(rows):
        if index == header_index:
            continue
        if label := _section_label(row):
            section = label
            lines.append(label)
            sections.append(label)
            continue
        names = headers or [""] * columns
        cells = [f"{h}: {v}" if h else v for h, v in zip(names, row, strict=True) if v]
        line = " | ".join(cells)
        lines.append(f"{section} – {line}" if section else line)

    if header_index is not None and len(lines) == 0:
        lines.append(" | ".join(c for c in rows[header_index] if c))
    return lines, sections, _TableContext(columns, headers, section)


# --------------------------------------------------------------------------- #
# Borderless tables (columns printed as separate narrow text blocks)
# --------------------------------------------------------------------------- #
def _group_rows(lines: list[_Line]) -> list[list[_Line]]:
    ordered = sorted(lines, key=lambda ln: (ln.y_center, ln.x0))
    heights = sorted(ln.y1 - ln.y0 for ln in ordered)
    tolerance = heights[len(heights) // 2] * 0.5
    rows: list[list[_Line]] = []
    for line in ordered:
        if rows and line.y_center - rows[-1][0].y_center <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [sorted(row, key=lambda ln: ln.x0) for row in rows]


def _merge_header_continuations(rows: list[list[_Line]]) -> list[list[_Line]]:
    """Join a header printed on two lines ("Başarı" / "Notu" -> "Başarı Notu").

    Only a single short line directly under the header row, inside exactly
    one header column, is joined. Section labels ("GÜZ YARIYILI") are not.
    """
    if len(rows) < 3 or len(rows[0]) < 2 or _has_digit([ln.text for ln in rows[0]]):
        return rows
    header = list(rows[0])
    index = 1
    while index < len(rows) and len(rows[index]) == 1:
        line = rows[index][0]
        owners = [h for h in header if line.x0 < h.x1 and line.x1 > h.x0]
        if (
            len(owners) != 1
            or _DIGIT.search(line.text)
            or line.text.isupper()
            or _SECTION_WORDS.search(line.text)
        ):
            break
        owner = owners[0]
        header[header.index(owner)] = _Line(
            min(owner.x0, line.x0),
            owner.y0,
            max(owner.x1, line.x1),
            line.y1,
            f"{owner.text} {line.text}",
        )
        index += 1
    return [header, *rows[index:]]


def _row_cells(rows: list[list[_Line]]) -> list[list[str]]:
    """Place each line into a column using the x positions of the widest row."""
    anchors = max(rows, key=len)
    columns = len(anchors)
    cells: list[list[str]] = []
    for row in rows:
        out = [""] * columns
        if (
            len(row) == 1
            and columns >= 2
            and not any(row[0].x0 < a.x1 and row[0].x1 > a.x0 for a in anchors[1:])
        ):
            # A lone line in the first column: a section label spans the row.
            out = [row[0].text] * columns
        else:
            for line in row:
                centre = (line.x0 + line.x1) / 2
                column = min(
                    range(columns),
                    key=lambda c: abs(centre - (anchors[c].x0 + anchors[c].x1) / 2),
                )
                out[column] = f"{out[column]} {line.text}".strip()
        cells.append(out)
    return cells


def _vertical_gap(a: _Block, b: _Block) -> float:
    return max(a.y0, b.y0) - min(a.y1, b.y1)  # negative when they overlap


_TABLE_COLUMN_GAP_PT = 24.0  # justified text has ~10 pt between words, tables far more


def _split_row_block(block: _Block) -> list[_Block]:
    """PyMuPDF sometimes puts a whole table row into one block whose lines sit
    side by side ("Ders kayıtları   14.09.2026   18.09.2026"). Split such a
    block into one block per cell; ordinary paragraphs are returned as is.

    Justified text can also come out as words side by side, so a block is only
    split when every gap between side-by-side lines is column-sized.
    """
    bands: list[list[_Line]] = []
    for line in sorted(block.lines, key=lambda ln: ln.y_center):
        band = bands[-1] if bands else None
        if band and line.y_center - band[0].y_center <= 0.5 * (line.y1 - line.y0):
            band.append(line)
        else:
            bands.append([line])
    multi = [sorted(b, key=lambda ln: ln.x0) for b in bands if len(b) > 1]
    if not multi:
        return [block]
    for band in multi:
        gaps = [right.x0 - left.x1 for left, right in pairwise(band)]
        if min(gaps) < _TABLE_COLUMN_GAP_PT:
            return [block]
    return [_Block.from_lines([line]) for line in block.lines]


def _side_by_side_tables(blocks: list[_Block], page_width: float) -> list[_Block]:
    """Turn borderless tables into rows of cells.

    Handles both layouts PyMuPDF produces: one block per column (cells stacked
    in a narrow block) and one block per row (cells side by side). Blocks that
    do not end up in a table are returned unchanged, so normal paragraphs are
    never touched.
    """
    originals = blocks
    blocks, origin = [], []
    for index, block in enumerate(originals):
        pieces = [block] if block.is_table else _split_row_block(block)
        blocks.extend(pieces)
        origin.extend([index] * len(pieces))

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
        return originals

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
    result: list[_Block] = []
    for index, block in enumerate(originals):
        own = [i for i, o in enumerate(origin) if o == index]
        if not used.intersection(own):
            result.append(block)  # untouched: keep the original paragraph block
        else:
            result.extend(blocks[i] for i in own if i not in used)
    for group in tables:
        lines = [ln for i in group for ln in blocks[i].lines]
        merged = _Block.from_lines(lines)
        merged.table_rows = _row_cells(_merge_header_continuations(_group_rows(lines)))
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


def _page_text(
    page: pymupdf.Page,
    raw: dict[str, Any],
    body_size: float,
    table_context: _TableContext | None,
) -> tuple[PageText, _TableContext | None]:
    blocks = _page_blocks(page, raw, body_size)
    headings = [
        ln.text for b in blocks if not b.is_table for ln in b.lines if ln.is_heading
    ]
    context = table_context
    seen_table = False
    for block in blocks:
        if block.table_rows is None:
            continue
        seen_table = True
        lines, sections, context = _render_table(block.table_rows, context)
        # One paragraph per row: rows are never glued together by the cleaner
        # and the chunker prefers to cut between rows.
        block.rendered = "\n\n".join(lines)
        headings.extend(sections)
    page_text = PageText(
        page_number=page.number + 1,
        text="\n\n".join(b.text for b in blocks if b.text),
        headings=tuple(headings),
    )
    # A table only continues onto the next page from a page that had a table.
    return page_text, context if seen_table else None


def extract_pages(file_path: Path) -> list[PageText]:
    """Extract text per page, keeping real 1-based page numbers.

    Text blocks are separated by a blank line. Table rows become
    self-contained lines (``Bölüm – Sütun: değer | Sütun: değer``) so that
    column headers and section labels are never lost. Empty pages are kept
    with empty text so page numbering and statistics stay correct.

    Any unexpected PyMuPDF failure while reading pages is reported as
    ``UnreadablePDFError`` (HTTP 422) instead of an uncontrolled 500.
    """
    path = Path(file_path)
    doc = _open_pdf(path)
    try:
        result = _extract_open_document(doc)
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise UnreadablePDFError(
            f"PyMuPDF failed while reading {path}: {exc!r}"
        ) from exc
    finally:
        doc.close()

    if all(page.is_empty for page in result):
        raise NoExtractableTextError(f"No text layer found in {path}")

    logger.info(
        "Extracted %s pages (%s empty, %s with tables) from %s",
        len(result),
        sum(p.is_empty for p in result),
        sum(" | " in p.text for p in result),
        path.name,
    )
    return result


def _extract_open_document(doc: pymupdf.Document) -> list[PageText]:
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
    context: _TableContext | None = None
    for page, raw in zip(doc, raw_pages, strict=True):
        # Per-page body size copes with pages scaled differently.
        body_size = _body_font_size([raw]) or doc_body_size
        page_text, context = _page_text(page, raw, body_size, context)
        result.append(page_text)
    return result
