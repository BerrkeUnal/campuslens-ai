from pathlib import Path

from backend.app.rag.pdf_extractor import PageText, extract_pages
from backend.app.rag.text_cleaner import (
    clean_page_text,
    clean_pages,
    remove_repeated_margin_lines,
)

HEADER = "Örnek Üniversitesi Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği"


def test_repeated_header_and_footer_removed_from_three_pages() -> None:
    pages = [
        f"{HEADER}\n\nMADDE {i} – İçerik {i} burada.\n\nSayfa {i} / 3"
        for i in range(1, 4)
    ]
    cleaned = remove_repeated_margin_lines(pages)
    for i, text in enumerate(cleaned, start=1):
        assert HEADER not in text
        assert "Sayfa" not in text
        assert f"İçerik {i} burada." in text


def test_header_on_only_two_pages_is_kept() -> None:
    pages = [f"{HEADER}\nMetin A", f"{HEADER}\nMetin B", "Metin C"]
    assert remove_repeated_margin_lines(pages)[0].startswith(HEADER)


def test_repeated_line_in_body_is_not_removed() -> None:
    pages = [
        f"Başlık {c}\nGiriş {c}\nAçıklama {c}\nTEKRAR EDEN GÖVDE CÜMLESİ\nDetay {c}\nNot {c}\nSon {c}"
        for c in "ABCD"
    ]
    assert remove_repeated_margin_lines(pages) == pages


def test_header_removed_end_to_end_from_pdf(regulation_pdf: Path) -> None:
    pages = clean_pages(extract_pages(regulation_pdf))
    assert all(HEADER not in p.text for p in pages)
    assert all("Sayfa" not in p.text for p in pages)
    assert [p.page_number for p in pages] == [1, 2, 3, 4]


def test_standalone_page_number_removed_but_content_numbers_kept() -> None:
    text = "- 7 -\n\nMADDE 5 – Öğrenci en az 30 AKTS alır.\n\n12"
    cleaned = clean_page_text(text)
    assert cleaned == "MADDE 5 – Öğrenci en az 30 AKTS alır."


def test_number_in_middle_of_page_is_not_treated_as_page_number() -> None:
    text = "Başlık\nAçıklama\nDers kredisi:\n3\nToplam\nSon satır 1\nSon satır 2"
    assert "\n3" in clean_page_text(text) or " 3 " in clean_page_text(text)


def test_safe_dehyphenation() -> None:
    assert clean_page_text("öğren-\nci kaydı") == "öğrenci kaydı"
    assert clean_page_text("Eğitim-\nÖğretim yılı") == "Eğitim-Öğretim yılı"
    assert clean_page_text("2025-\n2026 yılı") == "2025-2026 yılı"


def test_whitespace_and_paragraphs_normalised() -> None:
    text = "Birinci   paragraf\nsatır devamı.\n\n\n\nİkinci\u00a0paragraf."
    assert clean_page_text(text) == "Birinci paragraf satır devamı.\n\nİkinci paragraf."


def test_list_items_keep_their_own_line() -> None:
    text = "Aşağıdaki şartlar aranır:\na) Devam şartını sağlamak,\nb) Harç ödemek."
    assert (
        clean_page_text(text)
        == "Aşağıdaki şartlar aranır:\na) Devam şartını sağlamak,\nb) Harç ödemek."
    )


def test_numbered_line_wrap_is_joined() -> None:
    assert (
        clean_page_text("Öğrenci\n3. yarıyılda staj yapar.")
        == "Öğrenci 3. yarıyılda staj yapar."
    )


def test_turkish_characters_and_ligatures() -> None:
    text = "ÇĞİÖŞÜ çğıöşü \ufb01nal o\u0308ğrenci\u00ad"
    assert clean_page_text(text) == "ÇĞİÖŞÜ çğıöşü final öğrenci"


def test_empty_page_cleans_to_empty_string() -> None:
    assert clean_page_text("  \n\n 3 \n") == ""
    assert clean_pages([PageText(1, "")])[0].text == ""


def test_known_footer_removed_from_short_last_page() -> None:
    footer = "Vadi Kampüs: Ayazağa Mah. Azerbaycan Cad. No: 3H Sarıyer, İstanbul"
    body = "\n".join(f"Gövde satırı {i}." for i in range(10))
    pages = [f"{body}\n{footer}" for _ in range(5)]
    pages.append(f"MADDE 39 – Bu Yönetmelik yürürlüğe girer.\n{footer}")
    cleaned = remove_repeated_margin_lines(pages)
    assert all(footer not in text for text in cleaned)
    assert cleaned[-1] == "MADDE 39 – Bu Yönetmelik yürürlüğe girer."


def test_table_rows_keep_their_own_lines() -> None:
    text = "Başarı | Katsayı\nAA | 4,00\nBA | 3,50"
    assert clean_page_text(text) == text


def test_table_row_paragraphs_are_not_glued_together() -> None:
    text = "Güz Yarıyılı\n\nGüz Yarıyılı – Etkinlik: Ders kayıtları | Tarih: 14.09.2026\n\nGüz Yarıyılı – Etkinlik: Final"
    assert clean_page_text(text) == text
