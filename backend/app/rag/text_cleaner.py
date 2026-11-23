"""Text cleaning for extracted PDF pages.

Owner: Person 3 (document processing).

Two levels:
* ``remove_repeated_margin_lines`` works across the whole document and drops
  header/footer lines repeated at the same position (top/bottom) of many pages.
* ``clean_page_text`` works on one page: unicode fixes, page-number lines,
  safe de-hyphenation, paragraph rebuilding and whitespace normalisation.

``clean_pages`` runs both in the correct order. The goal is to remove noise
without deleting meaningful content, so every rule is conservative.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.rag.pdf_extractor import PageText

# Number of non-empty lines at the top/bottom of a page treated as margin zone.
MARGIN_LINES = 3
MIN_REPEAT_PAGES = 3
MIN_REPEAT_RATIO = 0.4

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_SPACE_CHARS = dict.fromkeys(
    map(
        ord,
        "\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
        "\u2009\u200a\u202f\u205f\u3000\t\f\v",
    ),
    " ",
)
_REMOVE_CHARS = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)

_TR_LOWER = "a-zçğıöşüâîû"
_TR_LETTER = "A-Za-zÇĞİIÖŞÜçğıöşüÂÎÛâîû"

# A line that is only a page number: "12", "- 12 -", "Sayfa 3", "3 / 12", "Page 2 of 9"
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:"
    r"[-–—]\s*\d{1,3}\s*[-–—]"  # - 12 -
    r"|\d{1,3}"  # 12  (4-digit years such as 2025 are kept)
    r"|\d{1,4}\s*/\s*\d{1,4}"
    r"|(?:sayfa|page|sf\.?|s\.)\s*:?\s*\d{1,4}(?:\s*(?:/|of|'?(?:den|dan|ten|tan))\s*\d{1,4})?"
    r")\s*$",
    re.IGNORECASE,
)

# Lines that always start a new logical item (never glued to the previous line).
_LINE_START_MARKER = re.compile(
    r"^(?:"
    r"\(\d{1,3}\)"  # (1)
    r"|[-•▪●◦*–]\s"  # bullets
    r"|MADDE\s+\d+|Madde\s+\d+|GEÇİCİ\s+MADDE"
    r"|(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)\s+BÖLÜM"
    r")"
)
# "1." / "a)" start a list item only after a line that ended a sentence/intro;
# otherwise "Öğrenci" + "33. dönemde" is a normal line wrap.
_ENUM_MARKER = re.compile(r"^(?:\d{1,3}[.)]|[a-zçğıöşü][)])\s")
_ENDS_CLAUSE = (".", ":", ";", "!", "?", ",")  # list items in regulations end with ","
_LINE_END_HYPHEN = re.compile(rf"[{_TR_LETTER}0-9]-$")
_STARTS_LOWER = re.compile(rf"^[{_TR_LOWER}]")
_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")


def normalize_unicode(text: str) -> str:
    """NFC-normalise and fix PDF artefacts while keeping Turkish characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for lig, plain in _LIGATURES.items():
        text = text.replace(lig, plain)
    text = text.translate(_SPACE_CHARS).translate(_REMOVE_CHARS)
    # NFC composes e.g. "o" + U+0308 into "ö"; it never changes Turkish letters.
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace to single spaces. Used for hashing and matching."""
    return " ".join(normalize_unicode(text).split())


def _margin_key(line: str) -> str:
    """Key for comparing header/footer lines: digits are masked so that
    'Sayfa 3 / 12' and 'Sayfa 4 / 12' count as the same footer."""
    return re.sub(r"\d+", "#", normalize_whitespace(line).casefold())


def _margin_indexes(
    lines: list[str], margin: int, *, adaptive: bool = True
) -> tuple[list[int], list[int]]:
    """Indexes of the top/bottom margin lines.

    With ``adaptive`` the zone shrinks to a third of short pages, so body text
    of short pages never *creates* a header/footer pattern. Removal uses the
    full zone, so a known footer is still removed from a short last page.
    """
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    zone = max(1, min(margin, len(non_empty) // 3)) if adaptive else margin
    if not non_empty:
        return [], []
    return non_empty[:zone], non_empty[-zone:]


def remove_repeated_margin_lines(
    page_texts: list[str],
    *,
    margin_lines: int = MARGIN_LINES,
    min_repeat_pages: int = MIN_REPEAT_PAGES,
    min_repeat_ratio: float = MIN_REPEAT_RATIO,
) -> list[str]:
    """Remove header/footer lines repeated in the same zone on many pages.

    A line is removed only if it sits in the top or bottom ``margin_lines``
    non-empty lines of a page and the same (digit-masked) line appears in that
    same zone on at least ``min_repeat_pages`` pages and ``min_repeat_ratio``
    of non-empty pages. Body text is never touched.
    """
    split_pages = [text.split("\n") for text in page_texts]
    non_empty_pages = sum(1 for text in page_texts if text.strip())
    threshold = max(min_repeat_pages, int(non_empty_pages * min_repeat_ratio + 0.999))

    counts: Counter[tuple[str, str]] = Counter()
    for lines in split_pages:
        top, bottom = _margin_indexes(lines, margin_lines)
        seen = {("top", _margin_key(lines[i])) for i in top}
        seen |= {("bottom", _margin_key(lines[i])) for i in bottom}
        counts.update(seen)  # count each key once per page

    repeated = {key for key, count in counts.items() if count >= threshold and key[1]}
    if not repeated:
        return list(page_texts)

    cleaned: list[str] = []
    for lines in split_pages:
        top, bottom = _margin_indexes(lines, margin_lines, adaptive=False)
        drop = {i for i in top if ("top", _margin_key(lines[i])) in repeated}
        drop |= {i for i in bottom if ("bottom", _margin_key(lines[i])) in repeated}
        cleaned.append("\n".join(line for i, line in enumerate(lines) if i not in drop))
    return cleaned


def _remove_page_number_lines(lines: list[str]) -> list[str]:
    """Drop standalone page-number lines in the page margins only.

    Restricting to margins protects numbers that are real content, such as a
    lone value in a table of the academic calendar.
    """
    top, bottom = _margin_indexes(lines, 2)
    drop = {i for i in (*top, *bottom) if _PAGE_NUMBER_LINE.match(lines[i])}
    return [line for i, line in enumerate(lines) if i not in drop]


def _join_paragraph_lines(lines: list[str]) -> str:
    """Rebuild one paragraph from PDF lines.

    * ``öğren-`` + ``ci`` -> ``öğrenci`` only when the next line starts with a
      lowercase letter; ``Eğitim-`` + ``Öğretim`` -> ``Eğitim-Öğretim``.
    * Lines starting with list/article markers and table rows ("AA | 4,00")
      keep their own line.
    """
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
        elif _LINE_END_HYPHEN.search(out):
            out = out[:-1] + line if _STARTS_LOWER.match(line) else out + line
        elif (
            _LINE_START_MARKER.match(line)
            or (_ENUM_MARKER.match(line) and out.endswith(_ENDS_CLAUSE))
            or " | " in line  # table row from the extractor
            or " | " in out.rsplit("\n", 1)[-1]
        ):
            out = f"{out}\n{line}"
        else:
            out = f"{out} {line}"
    return out


def clean_page_text(text: str) -> str:
    """Clean a single page of text. Returns ``""`` for pages without content.

    Paragraphs are separated by a blank line (``\\n\\n``); list items and
    article lines inside a paragraph keep a single ``\\n``.
    """
    text = normalize_unicode(text)
    lines = [line.rstrip() for line in text.split("\n")]
    lines = _remove_page_number_lines(lines)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append(_join_paragraph_lines(current))
            current = []
    if current:
        paragraphs.append(_join_paragraph_lines(current))

    cleaned = "\n\n".join(
        _MULTI_SPACE.sub(" ", p).strip() for p in paragraphs if p.strip()
    )
    return _MULTI_BLANK.sub("\n\n", cleaned).strip()


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """Document-level cleaning: header/footer removal, then per-page cleaning.

    Page numbers and order are preserved; empty pages stay in the list with
    empty text so that statistics remain correct.
    """
    without_margins = remove_repeated_margin_lines([p.text for p in pages])
    return [
        replace(page, text=clean_page_text(text))
        for page, text in zip(pages, without_margins, strict=True)
    ]
