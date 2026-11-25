"""Page-aware, token-based chunking.

Owner: Person 3 (document processing).

Rules (from the shared project contract):
* 400 tokens per chunk, 60 tokens overlap, counted with the embedding model's
  tokenizer (intfloat/multilingual-e5-small).
* A chunk never crosses a page boundary, and the overlap never comes from
  another page.
* Cut at the nearest previous natural boundary: paragraph > sentence > word >
  token. A single paragraph longer than the limit is split by sentence, then by
  token.
* ``chunk_id = {document_id}-p{page_number}-c{page_chunk_index}``; the same
  input and settings always produce the same ids and texts.
* ``text`` holds only cleaned source text. The ``passage: `` prefix is added at
  embedding time by Person 4, never stored here.
* Output is the shared ``ChunkRecord`` schema (backend.app.schemas.retrieval).
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from backend.app.core.config import get_settings
from backend.app.rag.text_cleaner import normalize_whitespace
from backend.app.schemas.retrieval import ChunkRecord

if TYPE_CHECKING:
    from backend.app.rag.pdf_extractor import PageText

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 60


def compute_content_hash(text: str) -> str:
    """SHA-256 of whitespace/unicode-normalised text (for Person 4's payload)."""
    return hashlib.sha256(normalize_whitespace(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #
class TokenCounter(Protocol):
    name: str

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token, without special tokens."""

    def count(self, text: str) -> int: ...


class HFTokenCounter:
    """Exact token counts with the embedding model's own tokenizer."""

    def __init__(self, model_name: str | None = None) -> None:
        from tokenizers import Tokenizer

        model_name = model_name or get_settings().embedding_model
        self.name = model_name
        self._tokenizer = Tokenizer.from_pretrained(model_name)
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        return [(int(s), int(e)) for s, e in enc.offsets]

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class ApproximateTokenCounter:
    """Offline fallback. Splits words into <=4 character pieces, which over-
    rather than under-estimates XLM-R tokens for Turkish, so chunks stay
    within the model's 512-token window."""

    name = "approximate"
    _pattern = re.compile(r"\w{1,4}|[^\w\s]", re.UNICODE)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in self._pattern.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in self._pattern.finditer(text))


@lru_cache(maxsize=1)
def get_default_token_counter() -> TokenCounter:
    try:
        counter = HFTokenCounter()
        logger.info("Chunker uses tokenizer of %s", counter.name)
        return counter
    except Exception as exc:  # noqa: BLE001 - offline/missing model must not block uploads
        logger.warning(
            "Tokenizer %s could not be loaded (%s); using approximate counter. "
            "Run once with internet access to cache the tokenizer.",
            get_settings().embedding_model,
            exc,
        )
        return ApproximateTokenCounter()


# --------------------------------------------------------------------------- #
# Section titles
# --------------------------------------------------------------------------- #
_ARTICLE_LINE = re.compile(r"^(?:MADDE|Madde)\s+(\d+)\b")
_CHAPTER_LINE = re.compile(
    r"^(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)"
    r"\s+BÖLÜM\b"
)
_NOT_A_TITLE_END = (".", ",", ";", ":")


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    result, pos = [], 0
    for line in text.split("\n"):
        result.append((pos, line))
        pos += len(line) + 1
    return result


def _looks_like_title(line: str) -> bool:
    line = line.strip()
    return (
        2 <= len(line) <= 80
        and not line.endswith(_NOT_A_TITLE_END)
        and not _ARTICLE_LINE.match(line)
        and any(ch.isalpha() for ch in line)
    )


def find_section_markers(
    text: str,
    headings: tuple[str, ...] = (),
    carry_title: str | None = None,
) -> list[tuple[int, str]]:
    """Return ``(char_position, title)`` markers found reliably in a page.

    Sources, from most to least specific:
    1. Regulation articles: ``MADDE 12 –`` (+ the short title line above it).
    2. Chapters: ``İKİNCİ BÖLÜM`` (+ the title line below it).
    3. Lines the PDF styles as headings (bold/larger), matched at a line start.
    ``carry_title`` is a heading left alone at the bottom of the previous page
    ("Yürürlük"), used when this page starts directly with "MADDE n".
    Nothing is guessed: if no marker exists the chunk gets ``None``.
    """
    markers: dict[int, str] = {}
    lines = _lines_with_offsets(text)

    non_empty = [(pos, line.strip()) for pos, line in lines if line.strip()]
    for idx, (pos, line) in enumerate(non_empty):
        if m := _ARTICLE_LINE.match(line):
            title_pos, title = pos, f"MADDE {m.group(1)}"
            # The article title sits on the previous line ("Amaç" / "MADDE 1 –").
            if idx > 0 and _looks_like_title(non_empty[idx - 1][1]):
                title_pos, prev = non_empty[idx - 1]
                title = f"MADDE {m.group(1)} - {prev}"
            elif idx == 0 and carry_title:
                title = f"MADDE {m.group(1)} - {carry_title}"
            markers[title_pos] = title
        elif _CHAPTER_LINE.match(line):
            title = line
            # "İKİNCİ BÖLÜM" alone on its line: the chapter name is the next line.
            if (
                _CHAPTER_LINE.sub("", line).strip() == ""
                and idx + 1 < len(non_empty)
                and _looks_like_title(non_empty[idx + 1][1])
            ):
                title = f"{line} - {non_empty[idx + 1][1]}"
            markers.setdefault(pos, title)

    line_starts = {
        normalize_whitespace(line): pos for pos, line in reversed(lines) if line.strip()
    }
    for heading in headings:
        key = normalize_whitespace(heading)
        pos = line_starts.get(key)
        if pos is not None and pos not in markers and _looks_like_title(key):
            markers[pos] = key

    return sorted(markers.items())


def _content_anchor(text: str, position: int, max_skip: int = 6) -> int:
    """Skip heading-only lines ("ÜÇÜNCÜ BÖLÜM", "Bütünleme sınavı") at the start
    of a chunk so the title reflects the article the content belongs to."""
    pos = position
    for _ in range(max_skip):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        line_end = text.find("\n", pos)
        line_end = len(text) if line_end == -1 else line_end
        line = text[pos:line_end]
        if not line or not (
            _looks_like_title(line) or _CHAPTER_LINE.match(line.strip())
        ):
            return pos
        pos = line_end
    return position


def _trailing_heading(page: PageText, text: str) -> str | None:
    """A heading left alone at the very end of a page, e.g. "Yürürlük"."""
    last = text.rstrip().rsplit("\n", 1)[-1].strip()
    if not last or not _looks_like_title(last) or _CHAPTER_LINE.match(last):
        return None
    if page.headings and normalize_whitespace(last) not in {
        normalize_whitespace(h) for h in page.headings
    }:
        return None  # only trust it when the PDF styles it as a heading
    return last


def _section_at(
    markers: list[tuple[int, str]], position: int, inherited: str | None
) -> str | None:
    current = inherited
    for pos, title in markers:
        if pos > position:
            break
        current = title
    return current


# --------------------------------------------------------------------------- #
# Splitting a single page
# --------------------------------------------------------------------------- #
_SENTENCE_END = ".!?;:…"
# Words that end with "." without ending the sentence ("(Dğş. Senato 2025-15)").
_ABBREVIATIONS = frozenset(
    {
        "dğş",
        "değ",
        "ek",
        "mülga",
        "vb",
        "vs",
        "bkz",
        "örn",
        "md",
        "mad",
        "no",
        "s",
        "sf",
        "prof",
        "doç",
        "dr",
        "öğr",
        "gör",
        "yrd",
        "arş",
        "uzm",
        "av",
        "t.c",
        "rg",
        "r.g",
        "mah",
        "cad",
        "sok",
        "apt",
        "bl",
        "tel",
        "st",
        "yy",
        "vd",
        "a.ş",
        "ltd",
        "şti",
    }
)


def _dot_is_not_sentence_end(text: str, dot_index: int) -> bool:
    """True for "Dğş.", "Prof.", "54." and single-letter initials."""
    start = dot_index
    while (
        start > 0 and not text[start - 1].isspace() and text[start - 1] not in "(\"'“‘"
    ):
        start -= 1
    word = text[start:dot_index].casefold()
    return (
        word in _ABBREVIATIONS or word.isdigit() or (len(word) == 1 and word.isalpha())
    )


# Amendment notes belong to the sentence before them: "... alınır. (Dğş. Senato 2025-15)"
_AMENDMENT_NOTE = re.compile(
    r"\s*\((?:Dğş|Değ|Değişik|Ek|Mülga|İptal)\b", re.IGNORECASE
)


def _line_before(text: str, end: int) -> str:
    return text[text.rfind("\n", 0, end) + 1 : end]


_PARAGRAPH, _SENTENCE, _WORD, _TOKEN = 3, 2, 1, 0


def _starts_lowercase(token_text: str) -> bool:
    stripped = token_text.lstrip().lstrip("▁")
    return bool(stripped) and stripped[0].islower()


def _boundary_priorities(text: str, offs: list[tuple[int, int]]) -> list[int]:
    """priority[i] = how natural it is to start a chunk at token i."""
    prio = [_PARAGRAPH] + [_TOKEN] * (len(offs) - 1)
    for i in range(1, len(offs)):
        prev_end, (start, end) = offs[i - 1][1], offs[i]
        token_text = text[start:end]
        # Sentencepiece offsets may include the leading space in the token.
        separator = (
            text[prev_end:start]
            + token_text[: len(token_text) - len(token_text.lstrip())]
        )
        if not any(ch.isspace() for ch in separator):
            continue  # inside a word
        last_char = text[prev_end - 1]
        if "\n" in separator and _looks_like_title(_line_before(text, prev_end)):
            # Never leave a heading ("Bütünleme sınavı") at the end of a chunk,
            # separated from its "MADDE n" line.
            prio[i] = _WORD
        elif "\n\n" in separator:
            prio[i] = _PARAGRAPH
        elif "\n" in separator or (
            last_char in _SENTENCE_END
            and last_char != ":"  # "Başlangıç: 14.09.2026" is not a sentence end
            and not _starts_lowercase(token_text)
            and not (last_char == "." and _dot_is_not_sentence_end(text, prev_end - 1))
            and not _AMENDMENT_NOTE.match(text, prev_end)
        ):
            # "19. dönemde" / "Dğş. Senato" / "54. Maddesi" are not sentence ends.
            prio[i] = _SENTENCE
        else:
            prio[i] = _WORD
    return prio


def _choose_end(prio: list[int], start: int, limit: int, n: int, min_end: int) -> int:
    """Largest natural boundary in [min_end, limit], preferring higher priority."""
    if limit >= n:
        return n
    for level in (_PARAGRAPH, _SENTENCE, _WORD):
        for i in range(limit, min_end - 1, -1):
            if prio[i] >= level:
                return i
    return limit  # no natural boundary: cut at token limit


def split_page_text(
    text: str,
    counter: TokenCounter,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[tuple[int, int, int]]:
    """Split one page into ``(char_start, char_end, new_content_start)`` spans.

    ``new_content_start`` is where the non-overlapping part begins; it is used
    to pick the section title that the chunk is actually about.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if not 0 <= overlap_tokens < chunk_size_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_size_tokens)")

    offs = counter.offsets(text)
    n = len(offs)
    if n == 0:
        return []

    prio = _boundary_priorities(text, offs)
    min_progress = max(chunk_size_tokens // 2, overlap_tokens + 1)
    spans: list[tuple[int, int, int]] = []
    start, new_start = 0, 0

    while start < n:
        limit = min(start + chunk_size_tokens, n)
        end = _choose_end(prio, start, limit, n, min(start + min_progress, limit))

        # Re-tokenising a substring can differ by a token at the edges; shrink
        # until the stored text really fits the limit.
        while True:
            char_start, char_end = offs[start][0], offs[end - 1][1]
            if (
                counter.count(text[char_start:char_end]) <= chunk_size_tokens
                or end - start <= 1
            ):
                break
            end -= 1

        spans.append((char_start, char_end, offs[new_start][0]))
        if end >= n:
            break

        start, new_start = _overlap_start(prio, start, end, overlap_tokens), end

    return spans


def _overlap_start(prio: list[int], start: int, end: int, overlap: int) -> int:
    """First token of the next chunk: ~``overlap`` tokens before ``end``.

    Prefers a sentence start within +/-25 % of the target so the overlap reads
    naturally, otherwise the first word start at/after the target.
    """
    if overlap == 0:
        return end
    target = max(end - overlap, start + 1)
    # Wide enough to reach the start of a table row or sentence, so the next
    # chunk does not begin in the middle of one.
    slack = max(1, overlap // 2)
    window = range(max(target - slack, start + 1), min(target + slack, end - 1) + 1)
    # Prefer the start of a paragraph / table row, then of a sentence.
    for level in (_PARAGRAPH, _SENTENCE):
        starts = [i for i in window if prio[i] >= level]
        if starts:
            return min(starts, key=lambda i: (abs(i - target), i))
    word_starts = [i for i in range(target, end) if prio[i] >= _WORD]
    return word_starts[0] if word_starts else end


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def chunk_pages(
    pages: list[PageText],
    document_id: str,
    document_title: str,
    source_url: str,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    *,
    token_counter: TokenCounter | None = None,
) -> list[ChunkRecord]:
    """Turn cleaned pages into ChunkRecords. Empty pages produce no chunks.

    ``chunk_id`` = ``{document_id}-p{page_number}-c{n}``, n starts at 1 on every page.
    """
    if not document_id:
        raise ValueError("document_id is required (UUID created by Person 6)")

    counter = token_counter or get_default_token_counter()
    records: list[ChunkRecord] = []
    inherited_section: str | None = None

    carry_title: str | None = None

    for page in sorted(pages, key=lambda p: p.page_number):
        text = page.text.strip()
        if not text:
            continue
        markers = find_section_markers(text, page.headings, carry_title)
        spans = split_page_text(text, counter, chunk_size_tokens, overlap_tokens)

        for index, (char_start, char_end, new_start) in enumerate(spans, start=1):
            chunk_text = text[char_start:char_end].strip()
            if not chunk_text:
                continue
            section = _section_at(
                markers, _content_anchor(text, new_start), inherited_section
            )
            if section is None:
                # Preamble before the first article (e.g. a cover table on page 1):
                # use the first section that starts inside this chunk.
                section = next(
                    (t for pos, t in markers if char_start <= pos < char_end), None
                )
            records.append(
                ChunkRecord(
                    chunk_id=f"{document_id}-p{page.page_number}-c{index}",
                    document_id=document_id,
                    document_title=document_title,
                    source_url=source_url,
                    page_number=page.page_number,
                    section_title=section,
                    text=chunk_text,
                )
            )
        if markers:
            inherited_section = markers[-1][1]
        carry_title = _trailing_heading(page, text)

    return records
