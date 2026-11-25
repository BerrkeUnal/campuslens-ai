"""Dense (E5) and sparse (BM25) text encoders for the vector search layer.

Dense vectors come from a multilingual E5 model loaded with
sentence-transformers; sparse vectors come from FastEmbed's ``Qdrant/bm25``
model with the Turkish Snowball stemmer. Both models are loaded lazily,
exactly once per ``TextEmbedder`` instance, and the default instance is shared
by the whole process.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from qdrant_client import models

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ApplicationError

logger = logging.getLogger(__name__)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
DEFAULT_BATCH_SIZE = 64
BM25_LANGUAGE = "turkish"
# Bump whenever prepare_bm25_text changes: existing indexes must be rebuilt.
BM25_PREPROCESSING_VERSION = "tr-fold-v1"

# Python's str.lower() maps "I" to "i" and "İ" to "i" + U+0307, which breaks
# Turkish words such as "İZİN" or "ISLAK". Map them explicitly first.
_TURKISH_CASE_MAP = str.maketrans({"I": "ı", "İ": "i"})
# Students often type without Turkish letters; fold both sides for BM25.
_ASCII_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
# Turkish suffixes after an apostrophe: "Erasmus'a", "2026'da", "AKTS'dir".
_APOSTROPHE_SUFFIX = re.compile(r"(?<=\w)['’‘`´]\w+")
_TURKISH_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")


class DenseModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any: ...


class SparseModel(Protocol):
    def embed(
        self,
        documents: Iterable[str],
        batch_size: int = ...,
    ) -> Iterable[Any]: ...

    def query_embed(self, query: str | Iterable[str]) -> Iterable[Any]: ...


DenseModelFactory = Callable[[str], DenseModel]
SparseModelFactory = Callable[[str], SparseModel]


class EmbeddingModelError(ApplicationError):
    """Raised when an embedding model cannot be loaded or used (HTTP 503)."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            code="embedding_unavailable",
            message=(
                "Arama modeli şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
            ),
            status_code=503,
        )
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


def _embedding_error(detail: str) -> EmbeddingModelError:
    # The API error handler only logs the error code, so keep the cause here.
    logger.error("Embedding failure: %s", detail)
    return EmbeddingModelError(detail)


# Text preparation -------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Return NFC-normalized text so equal strings embed identically."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """NFC-normalize and drop invisible format characters.

    Soft hyphens, zero-width spaces/joiners and BOMs (Unicode category Cf)
    are common in PDF text and split words for both tokenizers.
    """
    return "".join(
        char for char in normalize_text(text) if unicodedata.category(char) != "Cf"
    )


def has_searchable_text(text: str) -> bool:
    """True if the text contains anything besides whitespace/invisible marks."""
    return bool(text) and bool(clean_text(text).strip())


def turkish_lower(text: str) -> str:
    """Lowercase text using Turkish dotted and dotless i rules."""
    return normalize_text(text).translate(_TURKISH_CASE_MAP).lower()


def prepare_bm25_text(text: str) -> str:
    """Normalize text identically for BM25 indexing and querying.

    NFKC expands PDF ligatures ("ﬁ" -> "fi"), Turkish-aware lowercasing keeps
    "İZİN" intact, apostrophe suffixes are dropped and Turkish letters are
    folded to ASCII so "öğrenci", "ogrenci" and "OGRENCI" share one term.
    """
    text = unicodedata.normalize("NFKC", clean_text(text))
    text = _APOSTROPHE_SUFFIX.sub("", turkish_lower(text))
    return text.translate(_ASCII_FOLD)


def prepare_dense_query(question: str) -> str:
    """Clean a question for E5; lowercase it if it is written in all caps.

    E5's tokenizer is case-sensitive and all-caps questions embed poorly.
    Normal mixed-case questions are left untouched.
    """
    text = clean_text(question)

    if _is_all_caps(text):
        has_turkish = not _TURKISH_LETTERS.isdisjoint(text)
        text = turkish_lower(text) if has_turkish else text.lower()

    return text


def _is_all_caps(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    uppercase = sum(char.isupper() for char in letters)
    return len(letters) >= 4 and uppercase >= 0.7 * len(letters)


# Model loading ----------------------------------------------------------------


def _load_sentence_transformer(
    model_name: str,
    *,
    local_files_only: bool = True,
) -> DenseModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        device="cpu",
        local_files_only=local_files_only,
    )


def _load_sparse_model(
    model_name: str,
    *,
    local_files_only: bool = True,
) -> SparseModel:
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(
        model_name=model_name,
        language=BM25_LANGUAGE,
        local_files_only=local_files_only,
    )
    _load_folded_stopwords(model)
    return model


def _load_folded_stopwords(model: Any) -> None:
    """Replace FastEmbed's stop words with UTF-8 read, BM25-prepared ones.

    FastEmbed opens ``turkish.txt`` without an explicit encoding, so on
    Windows (cp1254) words such as "için" or "çok" are decoded wrongly and
    are not removed, while Linux removes them. The words must also go through
    ``prepare_bm25_text`` because the tokens they are compared with do.
    """
    inner = getattr(model, "model", None)
    model_dir = getattr(inner, "_model_dir", None)
    language = getattr(inner, "language", None)

    if inner is None or model_dir is None or language is None:
        logger.warning("Could not prepare BM25 stop words; using FastEmbed's")
        return

    stopwords_path = Path(model_dir) / f"{language}.txt"

    if not stopwords_path.is_file():
        logger.warning("BM25 stop word file not found: %s", stopwords_path)
        return

    words = stopwords_path.read_text(encoding="utf-8").splitlines()
    stopwords = {prepare_bm25_text(word).strip() for word in words}
    stopwords.discard("")
    inner.stopwords = stopwords
    # Used when FastEmbed rebuilds the model in worker processes (parallel=N).
    inner._custom_stopwords = stopwords


def _to_sparse_vector(embedding: Any) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(index) for index in embedding.indices],
        values=[float(value) for value in embedding.values],
    )


def _require_text(text: str) -> None:
    if not has_searchable_text(text):
        raise ValueError("Text to encode cannot be empty.")


class TextEmbedder:
    """Encodes passages and questions into dense and sparse vectors."""

    def __init__(
        self,
        dense_model_name: str,
        sparse_model_name: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        local_files_only: bool = True,
        dense_model_factory: DenseModelFactory | None = None,
        sparse_model_factory: SparseModelFactory | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.batch_size = batch_size

        self._dense_model_factory = dense_model_factory or (
            lambda name: _load_sentence_transformer(
                name,
                local_files_only=local_files_only,
            )
        )
        self._sparse_model_factory = sparse_model_factory or (
            lambda name: _load_sparse_model(
                name,
                local_files_only=local_files_only,
            )
        )

        self._dense_model: DenseModel | None = None
        self._sparse_model: SparseModel | None = None
        self._dense_dimension: int | None = None
        self._lock = threading.Lock()

    @property
    def dense_dimension(self) -> int:
        """Size of the dense vectors produced by the loaded model."""
        self._get_dense_model()
        assert self._dense_dimension is not None
        return self._dense_dimension

    def load(self) -> None:
        """Load both models eagerly, e.g. at application startup."""
        self._get_dense_model()
        self._get_sparse_model()

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode passages with the E5 ``passage:`` prefix."""
        if not texts:
            return []

        model = self._get_dense_model()
        prepared = [PASSAGE_PREFIX + clean_text(text) for text in texts]
        _warn_if_truncated(model, prepared)
        return self._encode_dense(model, prepared)

    def encode_query(self, question: str) -> list[float]:
        """Encode a user question with the E5 ``query:`` prefix."""
        _require_text(question)
        model = self._get_dense_model()
        return self._encode_dense(
            model,
            [QUERY_PREFIX + prepare_dense_query(question)],
        )[0]

    def encode_sparse(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Encode passages into BM25 term-frequency sparse vectors."""
        if not texts:
            return []

        model = self._get_sparse_model()
        prepared = [prepare_bm25_text(text) for text in texts]

        try:
            embeddings = list(model.embed(prepared, batch_size=self.batch_size))
        except Exception as exc:
            raise _embedding_error(
                f"Sparse encoding failed with {self.sparse_model_name}: {exc!r}",
            ) from exc

        if len(embeddings) != len(texts):
            raise _embedding_error(
                "Sparse model returned a different number of vectors "
                f"({len(embeddings)}) than inputs ({len(texts)}).",
            )

        return [_to_sparse_vector(embedding) for embedding in embeddings]

    def encode_sparse_query(self, question: str) -> models.SparseVector:
        """Encode a question for BM25 search; IDF is applied by Qdrant."""
        _require_text(question)
        model = self._get_sparse_model()

        try:
            embeddings = list(model.query_embed(prepare_bm25_text(question)))
        except Exception as exc:
            raise _embedding_error(
                f"Sparse query encoding failed with {self.sparse_model_name}: {exc!r}",
            ) from exc

        if len(embeddings) != 1:
            raise _embedding_error(
                f"Sparse model returned {len(embeddings)} query vectors, expected 1.",
            )

        return _to_sparse_vector(embeddings[0])

    def _encode_dense(self, model: DenseModel, texts: list[str]) -> list[list[float]]:
        try:
            raw = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise _embedding_error(
                f"Dense encoding failed with {self.dense_model_name}: {exc!r}",
            ) from exc

        vectors = np.asarray(raw, dtype=np.float32)
        expected_shape = (len(texts), self._dense_dimension)

        if vectors.shape != expected_shape:
            raise _embedding_error(
                f"Dense model returned shape {vectors.shape}, "
                f"expected {expected_shape}.",
            )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
            raise _embedding_error("Dense model returned an invalid vector.")

        return (vectors / norms).tolist()

    def _get_dense_model(self) -> DenseModel:
        if self._dense_model is not None:
            return self._dense_model

        with self._lock:
            if self._dense_model is None:
                logger.info("Loading dense embedding model %s", self.dense_model_name)
                model = self._create_model(
                    self._dense_model_factory,
                    self.dense_model_name,
                )
                self._dense_dimension = _detect_dimension(
                    model,
                    self.dense_model_name,
                )
                self._dense_model = model

        return self._dense_model

    def _get_sparse_model(self) -> SparseModel:
        if self._sparse_model is not None:
            return self._sparse_model

        with self._lock:
            if self._sparse_model is None:
                logger.info("Loading sparse model %s", self.sparse_model_name)
                self._sparse_model = self._create_model(
                    self._sparse_model_factory,
                    self.sparse_model_name,
                )

        return self._sparse_model

    @staticmethod
    def _create_model(factory: Callable[[str], Any], model_name: str) -> Any:
        try:
            return factory(model_name)
        except Exception as exc:
            raise _embedding_error(
                f"Could not load model {model_name}: {exc!r}",
            ) from exc


def _detect_dimension(model: DenseModel, model_name: str) -> int:
    # sentence-transformers 6 renamed the method; keep the old name as fallback.
    for attribute in (
        "get_embedding_dimension",
        "get_sentence_embedding_dimension",
    ):
        getter = getattr(model, attribute, None)

        if callable(getter):
            dimension = getter()

            if dimension:
                return int(dimension)

    try:
        probe = model.encode(
            [PASSAGE_PREFIX + "boyut"],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise _embedding_error(
            f"Could not detect the vector size of {model_name}: {exc!r}",
        ) from exc

    return int(np.asarray(probe).shape[-1])


def _warn_if_truncated(model: Any, texts: list[str]) -> None:
    """Log when passages exceed the model's token limit (tail is ignored)."""
    tokenizer = getattr(model, "tokenizer", None)
    max_length = getattr(model, "max_seq_length", None)

    if tokenizer is None or not max_length:
        return

    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
    except Exception:  # noqa: BLE001 - the warning is best effort only
        return

    too_long = sum(length > max_length for length in lengths)

    if too_long:
        logger.warning(
            "%d of %d passages exceed %d tokens; dense search ignores the rest. "
            "Chunks should be counted with the embedding model's tokenizer.",
            too_long,
            len(texts),
            max_length,
        )


_default_embedder: TextEmbedder | None = None
_default_embedder_lock = threading.Lock()


def get_embedder() -> TextEmbedder:
    """Return the process-wide embedder configured from Settings."""
    global _default_embedder

    if _default_embedder is None:
        with _default_embedder_lock:
            if _default_embedder is None:
                settings = get_settings()
                _default_embedder = TextEmbedder(
                    settings.embedding_model,
                    settings.sparse_model,
                )

    return _default_embedder


def encode_documents(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode_documents(texts)


def encode_query(question: str) -> list[float]:
    return get_embedder().encode_query(question)


def encode_sparse(texts: list[str]) -> list[models.SparseVector]:
    return get_embedder().encode_sparse(texts)


def encode_sparse_query(question: str) -> models.SparseVector:
    return get_embedder().encode_sparse_query(question)
