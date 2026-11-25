"""Small synthetic PDFs for document-processing tests (Person 3).

Built with PyMuPDF at test time, so no PDF copies are stored in git. Every
helper closes its document, which matters on Windows where an open handle
prevents deleting or renaming the file.
"""

from __future__ import annotations

import html
from pathlib import Path

import pymupdf

CALENDAR_TITLE = "2026-2027 Eğitim-Öğretim Yılı Akademik Takvimi"
CALENDAR_HEADERS = ("Etkinlik", "Başlangıç", "Bitiş")
_COLUMNS = (60.0, 330.0, 445.0, 560.0)


def _text(
    page: pymupdf.Page, rect: pymupdf.Rect, value: str, *, bold: bool = False
) -> None:
    weight = "bold" if bold else "normal"
    page.insert_htmlbox(
        rect, f"<p style='font-size:9px;font-weight:{weight}'>{html.escape(value)}</p>"
    )


def _ruled_row(
    page: pymupdf.Page, y: float, cells: tuple[str, ...], *, bold: bool = False
) -> None:
    for column, value in enumerate(cells):
        rect = pymupdf.Rect(_COLUMNS[column], y, _COLUMNS[column + 1], y + 20)
        page.draw_rect(rect, color=(0, 0, 0), width=0.7)
        _text(
            page,
            pymupdf.Rect(rect.x0 + 3, rect.y0 + 3, rect.x1 - 3, rect.y1 - 2),
            value,
            bold=bold,
        )


def _ruled_section(page: pymupdf.Page, y: float, label: str) -> None:
    rect = pymupdf.Rect(_COLUMNS[0], y, _COLUMNS[-1], y + 20)  # one merged cell
    page.draw_rect(rect, color=(0, 0, 0), width=0.7)
    _text(
        page,
        pymupdf.Rect(rect.x0 + 3, rect.y0 + 3, rect.x1 - 3, rect.y1 - 2),
        label,
        bold=True,
    )


def calendar_rows(semester: str, month: str, count: int) -> list[tuple[str, str, str]]:
    return [
        (
            f"{semester} etkinliği {i}: ders kaydı ve danışman onayı",
            f"{i:02d}.{month}",
            f"{i + 3:02d}.{month}",
        )
        for i in range(1, count + 1)
    ]


def build_ruled_calendar(
    path: Path, *, fall_rows: int = 18, spring_rows: int = 18
) -> Path:
    """Two-page academic calendar drawn as a ruled table.

    Page 1: header row, merged "Güz Yarıyılı" row, fall rows, merged
    "Bahar Yarıyılı" row, first half of spring rows. Page 2 continues the
    spring rows WITHOUT repeating the header or the section row.
    """
    fall = calendar_rows("Güz", "09.2026", fall_rows)
    spring = calendar_rows("Bahar", "02.2027", spring_rows)
    split = spring_rows // 2

    doc = pymupdf.open()
    try:
        page = doc.new_page(width=595, height=842)
        _text(page, pymupdf.Rect(60, 40, 560, 70), CALENDAR_TITLE, bold=True)
        y = 80.0
        _ruled_row(page, y, CALENDAR_HEADERS, bold=True)
        y += 20
        _ruled_section(page, y, "Güz Yarıyılı")
        y += 20
        for row in fall:
            _ruled_row(page, y, row)
            y += 20
        _ruled_section(page, y, "Bahar Yarıyılı")
        y += 20
        for row in spring[:split]:
            _ruled_row(page, y, row)
            y += 20

        page = doc.new_page(width=595, height=842)
        y = 60.0
        for row in spring[split:]:
            _ruled_row(page, y, row)
            y += 20
        doc.save(path)
    finally:
        doc.close()
    return path


def build_borderless_calendar(path: Path) -> Path:
    """Calendar printed without ruling lines: every cell is its own text block."""
    rows: list[tuple[str, ...]] = [
        CALENDAR_HEADERS,
        ("GÜZ YARIYILI",),
        ("Ders kayıtları", "14.09.2026", "18.09.2026"),
        ("Derslerin başlaması", "21.09.2026", "21.09.2026"),
        ("Ara sınavlar", "09.11.2026", "13.11.2026"),
        ("BAHAR YARIYILI",),
        ("Ders kayıtları", "08.02.2027", "12.02.2027"),
        ("Final sınavları", "31.05.2027", "11.06.2027"),
    ]
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=595, height=842)
        _text(page, pymupdf.Rect(60, 40, 560, 70), CALENDAR_TITLE, bold=True)
        y = 90.0
        for index, row in enumerate(rows):
            for column, value in enumerate(row):
                rect = pymupdf.Rect(
                    _COLUMNS[column] + 3, y, _COLUMNS[column] + 105, y + 14
                )
                _text(page, rect, value, bold=index == 0 or len(row) == 1)
            y += 16
        doc.save(path)
    finally:
        doc.close()
    return path


def build_senate_header_document(path: Path, pages: int = 4) -> Path:
    """Directive whose every page starts with the senate meeting table."""
    doc = pymupdf.open()
    try:
        for number in range(1, pages + 1):
            page = doc.new_page(width=595, height=842)
            _text(
                page,
                pymupdf.Rect(60, 30, 560, 60),
                "İSTİNYE ÜNİVERSİTESİ SENATO TOPLANTISI",
                bold=True,
            )
            y = 62.0
            for cells, bold in (
                (("TOPLANTI TARİHİ", "TOPLANTI SAAT", "OTURUM NO"), True),
                (("12/08/2026", "13.30", "22"), False),
            ):
                _ruled_row(page, y, cells, bold=bold)
                y += 20
            body = " ".join(
                f"MADDE {number} – ({i}) Öğrenci başvurusunu akademik takvimde belirtilen sürede yapar."
                for i in range(1, 8)
            )
            _text(page, pymupdf.Rect(60, 120, 560, 700), body)
        doc.save(path)
    finally:
        doc.close()
    return path


def build_text_pdf(
    path: Path, text: str = "MADDE 1 – Bu yönerge öğrencilere uygulanır."
) -> Path:
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=595, height=842)
        _text(page, pymupdf.Rect(60, 60, 540, 780), text)
        doc.save(path)
    finally:
        doc.close()
    return path


def encrypted_pdf_bytes(tmp_dir: Path) -> bytes:
    path = tmp_dir / "parolali.pdf"
    doc = pymupdf.open()
    try:
        doc.new_page().insert_text((72, 72), "gizli")
        doc.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw="1234",
            owner_pw="1234",
        )
    finally:
        doc.close()
    return path.read_bytes()


def image_only_pdf_bytes(tmp_dir: Path) -> bytes:
    """A PDF without a text layer, like a scanned document."""
    path = tmp_dir / "taranmis.pdf"
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0.5, 0.5, 0.5))
        doc.save(path)
    finally:
        doc.close()
    return path.read_bytes()


CORRUPT_PDF_BYTES = (
    b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n%%garbage\x00\x01\x02 no xref"
)
