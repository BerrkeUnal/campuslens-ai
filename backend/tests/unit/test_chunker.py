from itertools import pairwise
from pathlib import Path

import pytest

from backend.app.rag.chunker import (
    ApproximateTokenCounter,
    HFTokenCounter,
    _boundary_priorities,
    chunk_pages,
    compute_content_hash,
    find_section_markers,
    split_page_text,
)
from backend.app.rag.pdf_extractor import PageText, extract_pages
from backend.app.rag.text_cleaner import clean_pages
from backend.app.schemas.retrieval import ChunkRecord

DOC_ID = "3f6c2a1e-0000-4000-8000-000000000001"
REQUIRED = {
    "chunk_id",
    "document_id",
    "document_title",
    "source_url",
    "page_number",
    "section_title",
    "text",
}


def _sentences(n: int, prefix: str = "Cümle") -> str:
    return " ".join(
        f"{prefix} {i} öğrenci işleri bütünleme sınavı hakkında bilgi verir."
        for i in range(n)
    )


def _chunk(pages, counter, **kw):
    return chunk_pages(
        pages,
        DOC_ID,
        "Yönetmelik",
        "https://example.edu.tr/y.pdf",
        token_counter=counter,
        **kw,
    )


# --- mandatory scenarios ---------------------------------------------------- #
def test_no_chunk_spans_two_pages(tmp_path: Path, ws_counter, pdf_builder) -> None:
    pdf = pdf_builder(
        tmp_path / "iki.pdf",
        [
            {"body": f"<p>{_sentences(60, 'Bir')}</p>"},
            {"body": f"<p>{_sentences(60, 'İki')}</p>"},
        ],
    )
    pages = clean_pages(extract_pages(pdf))
    chunks = _chunk(pages, ws_counter)
    by_page = {p.page_number: p.text for p in pages}
    assert {c.page_number for c in chunks} == {1, 2}
    for chunk in chunks:
        assert chunk.text in by_page[chunk.page_number]
        other = "İki" if chunk.page_number == 1 else "Bir"
        assert other not in chunk.text.split()


def test_empty_page_produces_no_chunk_and_next_pages_processed(ws_counter) -> None:
    pages = [
        PageText(1, "Birinci sayfa metni."),
        PageText(2, ""),
        PageText(3, "Üçüncü sayfa metni."),
    ]
    chunks = _chunk(pages, ws_counter)
    assert [c.page_number for c in chunks] == [1, 3]
    assert [c.chunk_id for c in chunks] == [f"{DOC_ID}-p1-c1", f"{DOC_ID}-p3-c1"]


def test_long_paragraph_split_with_about_60_token_overlap(ws_counter) -> None:
    text = _sentences(120)  # one paragraph, ~1200 tokens
    spans = split_page_text(text, ws_counter, 400, 60)
    assert len(spans) >= 3
    for (s1, e1, _), (s2, _, _) in pairwise(spans):
        assert ws_counter.count(text[s1:e1]) <= 400
        overlap = ws_counter.count(text[s2:e1])
        assert 45 <= overlap <= 75, overlap


def test_paragraph_without_sentence_boundaries_split_at_token_limit(ws_counter) -> None:
    text = " ".join(f"kelime{i}" for i in range(1000))
    spans = split_page_text(text, ws_counter, 400, 60)
    counts = [ws_counter.count(text[s:e]) for s, e, _ in spans]
    assert counts[0] == 400
    assert ws_counter.count(text[spans[1][0] : spans[0][1]]) == 60


def test_turkish_characters_preserved_in_chunks(ws_counter) -> None:
    text = "Öğrencinin ÇĞİÖŞÜ çğıöşü karakterleri bozulmamalıdır."
    assert _chunk([PageText(1, text)], ws_counter)[0].text == text


def test_two_runs_give_identical_ids_and_texts(regulation_pdf: Path) -> None:
    counter = ApproximateTokenCounter()
    run1 = _chunk(clean_pages(extract_pages(regulation_pdf)), counter)
    run2 = _chunk(clean_pages(extract_pages(regulation_pdf)), counter)
    assert [c.chunk_id for c in run1] == [c.chunk_id for c in run2]
    assert [c.text for c in run1] == [c.text for c in run2]
    assert len({c.chunk_id for c in run1}) == len(run1)


# --- contract & metadata ---------------------------------------------------- #
def test_chunk_record_has_all_shared_fields(regulation_pdf: Path) -> None:
    assert set(ChunkRecord.model_fields) >= REQUIRED
    chunks = _chunk(
        clean_pages(extract_pages(regulation_pdf)), ApproximateTokenCounter()
    )
    per_page: dict[int, int] = {}
    for chunk in chunks:
        assert isinstance(chunk, ChunkRecord)
        data = chunk.model_dump()
        assert data.keys() >= REQUIRED
        assert data["text"].strip()
        assert not data["text"].startswith("passage:")
        per_page[chunk.page_number] = per_page.get(chunk.page_number, 0) + 1
        assert (
            data["chunk_id"]
            == f"{DOC_ID}-p{chunk.page_number}-c{per_page[chunk.page_number]}"
        )


def test_content_hash_ignores_whitespace_differences() -> None:
    assert len(compute_content_hash("a")) == 64
    assert compute_content_hash("Bütünleme  sınavı\n") == compute_content_hash(
        "Bütünleme sınavı"
    )
    assert compute_content_hash("a") != compute_content_hash("b")


def test_chunks_respect_token_limit(regulation_pdf: Path) -> None:
    counter = ApproximateTokenCounter()
    chunks = _chunk(clean_pages(extract_pages(regulation_pdf)), counter)
    assert all(0 < counter.count(c.text) <= 400 for c in chunks)


def test_short_paragraphs_are_merged(ws_counter) -> None:
    text = "\n\n".join(f"Kısa paragraf {i}." for i in range(10))
    assert len(_chunk([PageText(1, text)], ws_counter)) == 1


def test_section_title_from_regulation_article(regulation_pdf: Path) -> None:
    chunks = _chunk(
        clean_pages(extract_pages(regulation_pdf)), ApproximateTokenCounter()
    )
    titles = {c.page_number: c.section_title for c in chunks}
    assert titles[1] == "MADDE 1 - Amaç"
    assert titles[2] == "MADDE 20 - Bütünleme sınavı"
    assert all(
        c.section_title == "MADDE 30 - Kayıt yenileme"
        for c in chunks
        if c.page_number == 4
    )


def test_chunk_starting_with_chapter_gets_article_title(ws_counter) -> None:
    text = "ÜÇÜNCÜ BÖLÜM\n\nSınavlar\n\nBütünleme sınavı\n\nMADDE 20 – (1) Bütünleme sınavı yapılır."
    assert (
        _chunk([PageText(1, text)], ws_counter)[0].section_title
        == "MADDE 20 - Bütünleme sınavı"
    )


def test_section_title_is_none_when_not_found(ws_counter) -> None:
    assert (
        _chunk([PageText(1, "Başlıksız düz metin.")], ws_counter)[0].section_title
        is None
    )


def test_section_markers_chapter_and_heading() -> None:
    text = "İKİNCİ BÖLÜM\nSınavlar\n\nGenel Hükümler\n\nMetin."
    markers = find_section_markers(text, headings=("Genel Hükümler",))
    assert [t for _, t in markers] == ["İKİNCİ BÖLÜM - Sınavlar", "Genel Hükümler"]


def test_invalid_overlap_rejected(ws_counter) -> None:
    with pytest.raises(ValueError):
        split_page_text("a b c", ws_counter, 10, 10)


def test_real_e5_tokenizer_if_available() -> None:
    try:
        counter = HFTokenCounter()
    except Exception:  # noqa: BLE001
        pytest.skip("intfloat/multilingual-e5-small tokenizer not cached / offline")
    text = _sentences(150)
    chunks = _chunk([PageText(1, text)], counter)
    assert len(chunks) > 1
    assert all(counter.count(c.text) <= 400 for c in chunks)


# --- regressions found on a real regulation (İstinye Üniversitesi) ---------- #
def test_abbreviation_and_ordinal_dots_are_not_sentence_ends(ws_counter) -> None:
    text = (
        "Cümle bir burada biter. (Dğş. Senato 2025-15) Hakkında 2547 sayılı Kanunun "
        "54. Maddesi uygulanır. Yeni cümle başlar."
    )
    offs = ws_counter.offsets(text)
    prio = _boundary_priorities(text, offs)
    starts = {text[s:e]: p for (s, e), p in zip(offs, prio, strict=True)}
    assert starts["(Dğş."] < 2  # amendment note stays with the previous sentence
    assert starts["Senato"] < 2  # "Dğş." is an abbreviation
    assert starts["Maddesi"] < 2  # "54." is an ordinal
    assert starts["Yeni"] == 2  # real sentence start


def test_heading_is_not_left_at_end_of_chunk(ws_counter) -> None:
    body = " ".join(f"Kelime{i} metin." for i in range(150))
    text = f"Birinci madde\n\nMADDE 1 – {body}\n\nİkinci madde\n\nMADDE 2 – {body}"
    spans = split_page_text(text, ws_counter, 330, 30)
    for start, end, _ in spans:
        assert not text[start:end].rstrip().endswith("İkinci madde")


def test_heading_at_bottom_of_previous_page_is_carried(ws_counter) -> None:
    pages = [
        PageText(
            1,
            "MADDE 38 – (1) Eski yönetmelik kaldırılmıştır.\n\nYürürlük",
            ("Yürürlük",),
        ),
        PageText(2, "MADDE 39 – (1) Bu Yönetmelik yayımı tarihinde yürürlüğe girer."),
    ]
    chunks = _chunk(pages, ws_counter)
    assert chunks[-1].section_title == "MADDE 39 - Yürürlük"


def test_preamble_chunk_uses_first_section_inside_it(ws_counter) -> None:
    text = (
        "SENATO TOPLANTISI TOPLANTI TARİHİ 09/07/2025 OTURUM NO 15 KARAR METNİ EKİDİR VE ONAYLANMIŞTIR\n\n"
        "Amaç\n\nMADDE 1 – (1) Bu Yönetmeliğin amacı esasları düzenlemektir."
    )
    assert _chunk([PageText(1, text)], ws_counter)[0].section_title == "MADDE 1 - Amaç"


def test_chapter_with_title_on_same_line() -> None:
    markers = find_section_markers(
        "İKİNCİ BÖLÜM Eğitim-Öğretim ile İlgili Esaslar\n\nMetin."
    )
    assert markers == [(0, "İKİNCİ BÖLÜM Eğitim-Öğretim ile İlgili Esaslar")]
