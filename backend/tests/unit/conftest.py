"""Fixtures for document-processing tests (Person 3).

Small synthetic PDFs are built with PyMuPDF at test time, so no PDF copies are
stored in git.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest

from backend.app.rag.chunker import TokenCounter

HEADER = "Örnek Üniversitesi Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği"

PageSpec = dict  # {"body": html str, "header": bool, "footer": bool}


def build_pdf(path: Path, pages: list[PageSpec], *, header: str = HEADER) -> Path:
    """Write a PDF where each page has optional header/footer and HTML body."""
    doc = pymupdf.open()
    total = len(pages)
    for number, spec in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)
        if spec.get("header", True) and spec.get("body"):
            page.insert_htmlbox(
                pymupdf.Rect(50, 30, 545, 60),
                f"<p style='font-size:9px'>{html.escape(header)}</p>",
            )
        if spec.get("body"):
            page.insert_htmlbox(pymupdf.Rect(50, 70, 545, 780), spec["body"])
        if spec.get("footer", True) and spec.get("body"):
            page.insert_htmlbox(
                pymupdf.Rect(50, 790, 545, 815),
                f"<p style='font-size:9px'>Sayfa {number} / {total}</p>",
            )
    doc.save(path)
    doc.close()
    return path


def article(no: int, title: str, text: str) -> str:
    return f"<p><b>{html.escape(title)}</b></p><p><b>MADDE {no} –</b> {html.escape(text)}</p>"


class WhitespaceCounter:
    """Deterministic test tokenizer: one token per whitespace-separated word."""

    name = "whitespace-test"

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]

    def count(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def pdf_builder() -> Callable[..., Path]:
    return build_pdf


@pytest.fixture
def header_text() -> str:
    return HEADER


@pytest.fixture
def ws_counter() -> TokenCounter:
    return WhitespaceCounter()


@pytest.fixture
def regulation_pdf(tmp_path: Path) -> Path:
    """4 pages: header/footer on each content page, page 3 empty, long page 4."""
    long_text = " ".join(
        f"Öğrenci {i}. dönemde kayıt yeniler ve danışmanının onayı ile ders seçer."
        for i in range(1, 90)
    )
    amac = "Bu Yönetmeliğin amacı, Örnek Üniversitesinde eğitim-öğretime ilişkin esasları düzenlemektir."
    kapsam = "Bu Yönetmelik, öğrencilerin kayıt, sınav ve mezuniyet işlemlerini kapsar."
    butunleme = (
        "(1) Devamsızlık nedeniyle başarısız sayılan öğrenci bütünleme sınavına giremez. "
        "(2) Bütünleme sınavı, yarıyıl sonu sınavlarından sonra yapılır."
    )
    pages = [
        {"body": article(1, "Amaç", amac) + article(2, "Kapsam", kapsam)},
        {"body": article(20, "Bütünleme sınavı", butunleme)},
        {"body": ""},
        {"body": article(30, "Kayıt yenileme", long_text)},
    ]
    return build_pdf(tmp_path / "yonetmelik.pdf", pages)
