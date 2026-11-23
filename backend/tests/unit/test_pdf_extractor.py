from pathlib import Path

import pymupdf
import pytest

from backend.app.core.exceptions import ApplicationError
from backend.app.rag.pdf_extractor import (
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    PDFProcessingError,
    extract_pages,
    validate_pdf_bytes,
)


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
    assert "Başarı | Katsayı\nAA | 4,00\nBA | 3,50\nBB | 3,00" in text
    assert (
        text.index("tabloda belirtilmiştir")
        < text.index("AA | 4,00")
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
    assert "Ders kayıtları | 15.09.2025" in text
    assert "Final sınavları | 05.01.2026" in text


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
