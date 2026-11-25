from pathlib import Path

import pymupdf
import pytest

from backend.app.core.exceptions import ApplicationError
from backend.app.rag import pdf_extractor
from backend.app.rag.pdf_extractor import (
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    PDFProcessingError,
    UnreadablePDFError,
    extract_pages,
    validate_pdf_bytes,
)
from backend.app.rag.text_cleaner import clean_pages
from backend.tests import pdf_factory


def test_pages_have_real_one_based_numbers(regulation_pdf: Path) -> None:
    pages = extract_pages(regulation_pdf)
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    assert "MADDE 20" in pages[1].text


def test_empty_page_is_kept_but_empty(regulation_pdf: Path) -> None:
    pages = extract_pages(regulation_pdf)
    assert pages[2].is_empty
    assert not pages[3].is_empty  # pages after the empty one are still read


def test_turkish_characters_survive_extraction(tmp_path: Path, pdf_builder) -> None:
    text = "ÇĞİÖŞÜ çğıöşü — Öğrenci işleri, bütünleme sınavı, ılgıt ışık"
    pdf = pdf_builder(
        tmp_path / "tr.pdf",
        [{"body": f"<p>{text}</p>", "header": False, "footer": False}],
    )
    assert text in extract_pages(pdf)[0].text


def test_bold_lines_become_heading_hints(regulation_pdf: Path) -> None:
    assert "Bütünleme sınavı" in extract_pages(regulation_pdf)[1].headings


def test_corrupt_pdf_raises_controlled_error(tmp_path: Path) -> None:
    bad = tmp_path / "bozuk.pdf"
    bad.write_bytes(b"%PDF-1.7\n this is not really a pdf \x00\x01\x02")
    with pytest.raises(PDFProcessingError) as exc:
        extract_pages(bad)
    # Shared ApplicationError handler turns this into a Turkish JSON error.
    assert isinstance(exc.value, ApplicationError)
    assert exc.value.message and exc.value.status_code == 415


def test_non_pdf_with_pdf_extension_is_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "sahte.pdf"
    fake.write_text("Merhaba, ben aslında bir metin dosyasıyım.", encoding="utf-8")
    with pytest.raises(InvalidPDFError):
        extract_pages(fake)


def test_password_protected_pdf_raises_controlled_error(tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "parolali.pdf"
    doc.save(
        path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="1234", owner_pw="1234"
    )
    with pytest.raises(EncryptedPDFError) as exc:
        extract_pages(path)
    assert "parola" in exc.value.message


def test_pdf_without_text_layer_raises(tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "taranmis.pdf"
    doc.save(path)
    with pytest.raises(NoExtractableTextError):
        extract_pages(path)


def test_validate_bytes_rejects_empty_upload() -> None:
    with pytest.raises(InvalidPDFError):
        validate_pdf_bytes(b"")


# --- tables ------------------------------------------------------------------ #
def _html(
    page: pymupdf.Page, rect: tuple[float, float, float, float], body: str
) -> None:
    page.insert_htmlbox(pymupdf.Rect(*rect), f"<p style='font-size:11px'>{body}</p>")


def test_borderless_table_columns_become_rows(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _html(
        page,
        (70, 60, 520, 90),
        "a) Başarı notları ve katsayıları aşağıdaki tabloda belirtilmiştir:",
    )
    _html(page, (110, 100, 160, 120), "<b>Başarı</b>")
    _html(page, (170, 100, 230, 120), "<b>Katsayı</b>")
    _html(page, (110, 122, 160, 200), "AA<br>BA<br>BB")
    _html(page, (170, 122, 230, 200), "4,00<br>3,50<br>3,00")
    _html(page, (70, 230, 520, 260), "(4) Bir dersin geçer başarı notu CC'dir.")
    path = tmp_path / "tablo.pdf"
    doc.save(path)

    text = extract_pages(path)[0].text
    # Every row carries its column headers, so values never lose their labels.
    assert (
        "Başarı: AA | Katsayı: 4,00\n\nBaşarı: BA | Katsayı: 3,50\n\nBaşarı: BB | Katsayı: 3,00"
        in text
    )
    assert (
        text.index("tabloda belirtilmiştir")
        < text.index("Başarı: AA")
        < text.index("geçer başarı")
    )


def test_ruled_table_rows_are_kept_together(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    cells = [
        ["Etkinlik", "Tarih"],
        ["Ders kayıtları", "15.09.2025"],
        ["Final sınavları", "05.01.2026"],
    ]
    x = [70, 300, 520]
    for r, row in enumerate(cells):
        y0 = 100 + r * 25
        for c, value in enumerate(row):
            page.draw_rect(
                pymupdf.Rect(x[c], y0, x[c + 1], y0 + 25), color=(0, 0, 0), width=0.8
            )
            _html(page, (x[c] + 4, y0 + 5, x[c + 1] - 4, y0 + 22), value)
    path = tmp_path / "takvim.pdf"
    doc.save(path)

    text = extract_pages(path)[0].text
    assert "Etkinlik: Ders kayıtları | Tarih: 15.09.2025" in text
    assert "Etkinlik: Final sınavları | Tarih: 05.01.2026" in text


def test_two_column_body_text_is_not_turned_into_a_table(tmp_path: Path) -> None:
    left = " ".join(["Sol sütundaki uzun paragraf metni burada devam eder."] * 6)
    right = " ".join(["Sağ sütundaki başka bir paragraf burada yer alır."] * 6)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _html(page, (40, 60, 290, 400), left)
    _html(page, (305, 60, 555, 400), right)
    path = tmp_path / "iki_sutun.pdf"
    doc.save(path)

    text = extract_pages(path)[0].text
    assert " | " not in text
    assert "Sol sütundaki uzun paragraf" in text and "Sağ sütundaki başka" in text


# --- academic calendars: headers and section rows must survive ------------------ #


def test_ruled_calendar_rows_carry_headers_and_section(tmp_path: Path) -> None:
    pages = extract_pages(pdf_factory.build_ruled_calendar(tmp_path / "takvim.pdf"))
    first, second = pages[0].text, pages[1].text

    assert (
        "Güz Yarıyılı – Etkinlik: Güz etkinliği 1: ders kaydı ve danışman onayı"
        in first
    )
    assert "| Başlangıç: 01.09.2026 | Bitiş: 04.09.2026" in first
    assert "Bahar Yarıyılı – Etkinlik: Bahar etkinliği 1:" in first
    # Page 2 continues the table without header or section row: both are carried over.
    assert second.startswith("Bahar Yarıyılı – Etkinlik: Bahar etkinliği 10:")
    assert "Güz Yarıyılı" not in second


def test_every_calendar_row_names_its_semester(tmp_path: Path) -> None:
    pages = extract_pages(pdf_factory.build_ruled_calendar(tmp_path / "takvim.pdf"))
    rows = [ln for p in pages for ln in p.text.split("\n\n") if "etkinliği" in ln]
    assert len(rows) == 36
    for row in rows:
        semester = "Güz Yarıyılı" if "Güz etkinliği" in row else "Bahar Yarıyılı"
        assert row.startswith(f"{semester} – Etkinlik: ")
        assert "Başlangıç: " in row and "Bitiş: " in row


def test_calendar_section_rows_are_heading_hints(tmp_path: Path) -> None:
    pages = extract_pages(pdf_factory.build_ruled_calendar(tmp_path / "takvim.pdf"))
    assert {"Güz Yarıyılı", "Bahar Yarıyılı"} <= set(pages[0].headings)


def test_borderless_calendar_rows_carry_headers_and_section(tmp_path: Path) -> None:
    text = extract_pages(
        pdf_factory.build_borderless_calendar(tmp_path / "takvim.pdf")
    )[0].text
    assert (
        "GÜZ YARIYILI – Etkinlik: Ders kayıtları | Başlangıç: 14.09.2026 | Bitiş: 18.09.2026"
        in text
    )
    assert (
        "BAHAR YARIYILI – Etkinlik: Final sınavları | Başlangıç: 31.05.2027 | Bitiş: 11.06.2027"
        in text
    )


def test_two_line_header_is_joined_but_section_label_is_not() -> None:
    def line(x0: float, y0: float, text: str) -> pdf_extractor._Line:
        return pdf_extractor._Line(x0, y0, x0 + 40, y0 + 10, text)

    rows = [
        [line(100, 0, "Başarı"), line(200, 0, "Katsayı")],
        [line(102, 12, "Notu")],  # second line of the first header cell
        [line(100, 24, "AA"), line(200, 24, "4,00")],
    ]
    merged = pdf_extractor._merge_header_continuations(rows)
    assert [ln.text for ln in merged[0]] == ["Başarı Notu", "Katsayı"]

    rows[1] = [line(102, 12, "Güz Yarıyılı")]
    kept = pdf_extractor._merge_header_continuations(rows)
    assert [ln.text for ln in kept[0]] == ["Başarı", "Katsayı"]
    assert kept[1][0].text == "Güz Yarıyılı"


def test_repeated_senate_meeting_table_is_removed(tmp_path: Path) -> None:
    pages = clean_pages(
        extract_pages(pdf_factory.build_senate_header_document(tmp_path / "y.pdf"))
    )
    for page in pages:
        assert "TOPLANTI TARİHİ" not in page.text
        assert "SENATO TOPLANTISI" not in page.text
        assert page.text.startswith(f"MADDE {page.page_number} – (1)")


def test_unexpected_pymupdf_failure_is_a_controlled_422(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = pdf_factory.build_text_pdf(tmp_path / "bozuk-ic.pdf")

    def broken(*args, **kwargs):
        raise RuntimeError("code=2: zlib error: invalid stored block lengths")

    monkeypatch.setattr(pdf_extractor, "_page_blocks", broken)
    with pytest.raises(UnreadablePDFError) as exc:
        extract_pages(pdf)
    assert exc.value.status_code == 422
    assert "okunamadı" in exc.value.message
    pdf.unlink()  # the document handle was closed, so the file can be removed (Windows)


def test_justified_words_are_not_mistaken_for_table_cells() -> None:
    def line(x0: float, x1: float, text: str, y: float = 100) -> pdf_extractor._Line:
        return pdf_extractor._Line(x0, y, x1, y + 10, text)

    # Justified paragraph: PyMuPDF may return words side by side ~10 pt apart.
    justified = pdf_extractor._Block.from_lines(
        [line(70, 85, "(3)"), line(95, 150, "Üniversitede,"), line(160, 180, "ders")]
    )
    assert pdf_extractor._split_row_block(justified) == [justified]

    # Calendar row: cells are columns apart.
    row = pdf_extractor._Block.from_lines(
        [
            line(64, 102, "Ders kayıtları"),
            line(334, 366, "14.09.2026"),
            line(449, 481, "18.09.2026"),
        ]
    )
    assert len(pdf_extractor._split_row_block(row)) == 3
