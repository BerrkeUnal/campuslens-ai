"""Unit tests for the embedder; fake models keep them fast and offline."""

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.rag.embedder import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    EmbeddingModelError,
    TextEmbedder,
    _load_folded_stopwords,
    clean_text,
    has_searchable_text,
    prepare_bm25_text,
    prepare_dense_query,
    turkish_lower,
)

DIMENSION = 384


class FakeDenseModel:
    def __init__(self, dimension: int = DIMENSION, scale: float = 3.0) -> None:
        self.dimension = dimension
        self.scale = scale
        self.calls: list[dict] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences, **kwargs):
        self.calls.append({"sentences": list(sentences), **kwargs})
        # Deliberately not normalized, so the embedder must normalize.
        return np.full((len(sentences), self.dimension), self.scale, dtype=np.float32)


class FakeSparseModel:
    def __init__(self) -> None:
        self.embed_inputs: list[list[str]] = []
        self.query_inputs: list[str] = []

    def embed(self, documents, batch_size=256):
        documents = list(documents)
        self.embed_inputs.append(documents)
        return [
            SimpleNamespace(indices=np.array([index + 1]), values=np.array([0.5]))
            for index, _ in enumerate(documents)
        ]

    def query_embed(self, query):
        self.query_inputs.append(query)
        return [SimpleNamespace(indices=np.array([7, 9]), values=np.array([1, 1]))]


def make_embedder(dense=None, sparse=None, **kwargs) -> TextEmbedder:
    dense = dense or FakeDenseModel()
    sparse = sparse or FakeSparseModel()
    return TextEmbedder(
        "fake-dense",
        "fake-sparse",
        dense_model_factory=lambda name: dense,
        sparse_model_factory=lambda name: sparse,
        **kwargs,
    )


# Dense encoding -------------------------------------------------------------------


def test_encode_documents_uses_passage_prefix_and_batch_size() -> None:
    dense = FakeDenseModel()
    embedder = make_embedder(dense=dense, batch_size=8)

    vectors = embedder.encode_documents(["Staj süresi 40 iş günüdür."])

    assert dense.calls[0]["sentences"] == [
        PASSAGE_PREFIX + "Staj süresi 40 iş günüdür.",
    ]
    assert dense.calls[0]["batch_size"] == 8
    assert dense.calls[0]["normalize_embeddings"] is True
    assert len(vectors) == 1 and len(vectors[0]) == DIMENSION


def test_encode_query_uses_query_prefix_once_and_returns_unit_vector() -> None:
    dense = FakeDenseModel()
    embedder = make_embedder(dense=dense)

    vector = embedder.encode_query("Bütünleme sınavına kimler girebilir?")

    assert dense.calls[0]["sentences"] == [
        QUERY_PREFIX + "Bütünleme sınavına kimler girebilir?",
    ]
    assert len(vector) == DIMENSION
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_text_is_nfc_normalized_and_invisible_marks_removed() -> None:
    dense = FakeDenseModel()
    embedder = make_embedder(dense=dense)
    decomposed = "o\u0308g\u0306\u00adren\u200bci"  # NFD + soft hyphen + ZWSP

    embedder.encode_documents([decomposed])

    assert dense.calls[0]["sentences"] == [PASSAGE_PREFIX + "öğrenci"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Bütünleme sınavına kimler girebilir?",
            "Bütünleme sınavına kimler girebilir?",
        ),
        ("AKTS nedir?", "AKTS nedir?"),
        (
            "BÜTÜNLEME SINAVINA KİMLER GİREBİLİR?",
            "bütünleme sınavına kimler girebilir?",
        ),
        ("OGRENCI ISLERI NEREDE", "ogrenci isleri nerede"),
    ],
)
def test_only_all_caps_questions_are_lowercased_for_dense(
    question: str,
    expected: str,
) -> None:
    assert prepare_dense_query(question) == expected


def test_empty_document_list_does_not_load_models() -> None:
    created = []

    embedder = TextEmbedder(
        "fake-dense",
        "fake-sparse",
        dense_model_factory=lambda name: created.append(name) or FakeDenseModel(),
        sparse_model_factory=lambda name: created.append(name) or FakeSparseModel(),
    )

    assert embedder.encode_documents([]) == []
    assert embedder.encode_sparse([]) == []
    assert created == []


@pytest.mark.parametrize("question", ["", "   ", "\u200b\u200b\u200b"])
def test_empty_or_invisible_question_is_rejected(question: str) -> None:
    embedder = make_embedder()

    assert not has_searchable_text(question)

    with pytest.raises(ValueError):
        embedder.encode_query(question)

    with pytest.raises(ValueError):
        embedder.encode_sparse_query(question)


def test_models_are_loaded_only_once_even_from_many_threads() -> None:
    created = []
    barrier = threading.Barrier(8)

    def factory(name: str) -> FakeDenseModel:
        created.append(name)
        return FakeDenseModel()

    embedder = TextEmbedder(
        "fake-dense",
        "fake-sparse",
        dense_model_factory=factory,
        sparse_model_factory=lambda name: FakeSparseModel(),
    )

    def worker() -> None:
        barrier.wait()
        embedder.encode_query("ders kaydı")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    embedder.encode_documents(["metin"])

    assert created == ["fake-dense"]


def test_dense_dimension_comes_from_model() -> None:
    assert make_embedder(dense=FakeDenseModel(dimension=768)).dense_dimension == 768


def test_legacy_dimension_method_is_supported() -> None:
    class LegacyModel(FakeDenseModel):
        get_embedding_dimension = None

        def get_sentence_embedding_dimension(self) -> int:
            return 512

    assert make_embedder(dense=LegacyModel(dimension=512)).dense_dimension == 512


def test_wrong_vector_shape_raises_clear_error() -> None:
    class WrongShapeModel(FakeDenseModel):
        def encode(self, sentences, **kwargs):
            return np.ones((len(sentences), 10), dtype=np.float32)

    embedder = make_embedder(dense=WrongShapeModel())

    with pytest.raises(EmbeddingModelError, match="shape"):
        embedder.encode_query("soru")


def test_zero_vector_is_rejected() -> None:
    embedder = make_embedder(dense=FakeDenseModel(scale=0.0))

    with pytest.raises(EmbeddingModelError):
        embedder.encode_query("soru")


def test_model_load_failure_raises_503_and_logs_the_cause(caplog) -> None:
    def broken_factory(name: str):
        raise OSError("no internet")

    embedder = TextEmbedder(
        "missing-model",
        "fake-sparse",
        dense_model_factory=broken_factory,
        sparse_model_factory=lambda name: FakeSparseModel(),
    )

    with caplog.at_level(logging.ERROR), pytest.raises(EmbeddingModelError) as error:
        embedder.encode_query("soru")

    assert error.value.status_code == 503
    assert error.value.code == "embedding_unavailable"
    assert "missing-model" in str(error.value)
    assert "no internet" in caplog.text


def test_long_passages_log_a_truncation_warning(caplog) -> None:
    class TokenizingModel(FakeDenseModel):
        max_seq_length = 5

        @staticmethod
        def tokenizer(texts, **kwargs):
            return {"input_ids": [text.split() for text in texts]}

    embedder = make_embedder(dense=TokenizingModel())

    with caplog.at_level(logging.WARNING):
        embedder.encode_documents(
            ["kısa metin", "bu metin beş kelimeden çok daha uzun"]
        )

    assert "1 of 2 passages exceed 5 tokens" in caplog.text


# Sparse (BM25) encoding ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("İZİN BELGESİ", "izin belgesi"),
        ("ISLAK İMZA", "islak imza"),
        ("Öğrenci İşleri", "ogrenci isleri"),
        ("OGRENCI ISLERI", "ogrenci isleri"),
        ("IELTS ve IBAN", "ielts ve iban"),
        ("kâğıt, hâlâ", "kagit, hala"),
        ("Erasmus'a 2026’da AKTS'dir", "erasmus 2026 akts"),
        ("serti\ufb01ka", "sertifika"),
        ("öğ\u00adren\u200bci", "ogrenci"),
        ("I\u0307ZIN", "izin"),
    ],
)
def test_prepare_bm25_text_folds_turkish_variants(text: str, expected: str) -> None:
    assert prepare_bm25_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("İZİN BELGESİ", "izin belgesi"),
        ("ISLAK İMZA", "ıslak imza"),
        ("Iğdır İli", "ığdır ili"),
    ],
)
def test_turkish_lower_handles_dotted_and_dotless_i(text: str, expected: str) -> None:
    assert turkish_lower(text) == expected


def test_clean_text_drops_format_characters_only() -> None:
    assert clean_text("a\u00adb\u200bc\u200dd\ufeffe f") == "abcde f"


def test_sparse_encoding_uses_bm25_preparation_on_both_sides() -> None:
    sparse = FakeSparseModel()
    embedder = make_embedder(sparse=sparse)

    vectors = embedder.encode_sparse(["İZİN BELGESİ", "Staj"])
    query = embedder.encode_sparse_query("İzin nasıl alınır?")

    assert sparse.embed_inputs == [["izin belgesi", "staj"]]
    assert sparse.query_inputs == ["izin nasil alinir?"]
    assert [vector.indices for vector in vectors] == [[1], [2]]
    assert query.indices == [7, 9]
    assert query.values == [1.0, 1.0]


def test_sparse_vector_count_mismatch_is_detected() -> None:
    class DroppingSparseModel(FakeSparseModel):
        def embed(self, documents, batch_size=256):
            return super().embed(documents, batch_size)[:-1]

    embedder = make_embedder(sparse=DroppingSparseModel())

    with pytest.raises(EmbeddingModelError):
        embedder.encode_sparse(["bir", "iki"])


def test_stopwords_are_read_as_utf8_and_folded(tmp_path) -> None:
    (tmp_path / "turkish.txt").write_text("için\nçok\nmı\nve\n\n", encoding="utf-8")
    inner = SimpleNamespace(
        _model_dir=str(tmp_path),
        language="turkish",
        stopwords={"iÃ§in"},  # what cp1254 decoding produces
        _custom_stopwords=None,
    )

    _load_folded_stopwords(SimpleNamespace(model=inner))

    assert inner.stopwords == {"icin", "cok", "mi", "ve"}
    assert inner._custom_stopwords == inner.stopwords


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_embedder(batch_size=0)
