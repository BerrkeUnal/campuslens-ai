from unittest.mock import Mock

import pytest

from backend.app.rag import embedder


@pytest.mark.parametrize("local_files_only", [True, False])
def test_dense_loader_forwards_cache_policy(monkeypatch, local_files_only):
    factory = Mock()

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        factory,
    )

    result = embedder._load_sentence_transformer(
        "intfloat/multilingual-e5-small",
        local_files_only=local_files_only,
    )

    factory.assert_called_once_with(
        "intfloat/multilingual-e5-small",
        device="cpu",
        local_files_only=local_files_only,
    )

    assert result is factory.return_value


@pytest.mark.parametrize("local_files_only", [True, False])
def test_sparse_loader_forwards_cache_policy(monkeypatch, local_files_only):
    factory = Mock()
    stopwords = Mock()

    monkeypatch.setattr(
        "fastembed.SparseTextEmbedding",
        factory,
    )

    monkeypatch.setattr(
        embedder,
        "_load_folded_stopwords",
        stopwords,
    )

    result = embedder._load_sparse_model(
        "Qdrant/bm25",
        local_files_only=local_files_only,
    )

    factory.assert_called_once_with(
        model_name="Qdrant/bm25",
        language=embedder.BM25_LANGUAGE,
        local_files_only=local_files_only,
    )

    stopwords.assert_called_once_with(factory.return_value)

    assert result is factory.return_value


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"local_files_only": True}, True),
        ({"local_files_only": False}, False),
    ],
)
def test_text_embedder_propagates_cache_policy(monkeypatch, kwargs, expected):
    dense_factory = Mock()
    dense_factory.return_value.get_embedding_dimension.return_value = 384

    sparse_factory = Mock()

    monkeypatch.setattr(
        embedder,
        "_load_sentence_transformer",
        dense_factory,
    )

    monkeypatch.setattr(
        embedder,
        "_load_sparse_model",
        sparse_factory,
    )

    instance = embedder.TextEmbedder(
        "intfloat/multilingual-e5-small",
        "Qdrant/bm25",
        **kwargs,
    )

    instance.load()

    dense_factory.assert_called_once_with(
        "intfloat/multilingual-e5-small",
        local_files_only=expected,
    )

    sparse_factory.assert_called_once_with(
        "Qdrant/bm25",
        local_files_only=expected,
    )
