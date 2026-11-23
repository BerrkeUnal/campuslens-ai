"""Integration tests for the vector store with real E5/BM25 models.

Skipped by default so a plain ``pytest`` stays offline and fast. Enable with
``RUN_QDRANT_INTEGRATION=1`` (downloads the models on first run). Qdrant runs
in memory unless QDRANT_TEST_URL (e.g. ``http://localhost:6333``) points to
a server; every collection a test creates is deleted again.
"""

import asyncio
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from backend.app.core.config import get_settings
from backend.app.rag.embedder import TextEmbedder
from backend.app.rag.qdrant_store import (
    DENSE_VECTOR_NAME,
    PAYLOAD_FIELDS,
    SPARSE_VECTOR_NAME,
    CollectionNotFoundError,
    CollectionSchemaError,
    QdrantStore,
    QdrantVectorStore,
    VectorStoreUnavailableError,
    chunk_point_id,
    compute_content_hash,
    create_qdrant_client,
)
from backend.app.schemas.retrieval import (
    ChunkRecord,
    RetrievalMethod,
    RetrievedChunk,
)

RUN_INTEGRATION = os.environ.get("RUN_QDRANT_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="set RUN_QDRANT_INTEGRATION=1 to run tests with real models",
    ),
]

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_chunks.json"
REGULATION_DOC = "3f1c2a9e-8b7d-4c21-9f3e-1a2b3c4d5e6f"
INTERNSHIP_DOC = "a7b8c9d0-1e2f-4a3b-8c4d-5e6f7a8b9c0d"
ERASMUS_DOC = "c1d2e3f4-5a6b-4c7d-8e9f-0a1b2c3d4e5f"

RELEVANCE_CASES = [
    (
        "Devamsızlıktan kaldığım dersin bütünleme sınavına girebilir miyim?",
        f"{REGULATION_DOC}-p4-c0",
    ),
    ("Staj kaç iş günü sürüyor?", f"{INTERNSHIP_DOC}-p2-c0"),
    (
        "Erasmus başvurusu için not ortalaması kaç olmalı?",
        f"{ERASMUS_DOC}-p1-c0",
    ),
    ("Mezun olmak için kaç AKTS gerekir?", f"{REGULATION_DOC}-p9-c0"),
    # Typed without Turkish letters, and in capitals.
    ("devamsizliktan kaldim butunlemeye girebilir miyim", f"{REGULATION_DOC}-p4-c0"),
    ("ERASMUS BAŞVURUSU İÇİN NOT ORTALAMASI KAÇ?", f"{ERASMUS_DOC}-p1-c0"),
]


# Fixtures -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_embedder() -> TextEmbedder:
    settings = get_settings()
    embedder = TextEmbedder(settings.embedding_model, settings.sparse_model)
    embedder.load()
    return embedder


@pytest.fixture(scope="module")
def sample_chunk_records() -> list[ChunkRecord]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return [ChunkRecord.model_validate(item) for item in data]


@pytest.fixture
def qdrant_test_client() -> Iterator[QdrantClient]:
    client = create_qdrant_client(os.environ.get("QDRANT_TEST_URL", ":memory:"))
    yield client
    client.close()


@pytest.fixture
def make_store(
    qdrant_test_client: QdrantClient,
    real_embedder: TextEmbedder,
) -> Iterator[Callable[..., QdrantStore]]:
    """Build stores on uniquely named collections and always clean them up."""
    created: list[str] = []

    def factory(**kwargs) -> QdrantStore:
        name = f"test_chunks_{uuid.uuid4().hex[:12]}"
        created.append(name)
        return QdrantStore(
            client=qdrant_test_client,
            embedder=real_embedder,
            collection_name=name,
            **kwargs,
        )

    yield factory

    for name in created:
        if qdrant_test_client.collection_exists(name):
            qdrant_test_client.delete_collection(name)


@pytest.fixture
def vector_store(make_store) -> QdrantStore:
    return make_store()


@pytest.fixture
def indexed_vector_store(
    vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> QdrantStore:
    vector_store.create_collection()
    vector_store.upsert_chunks(sample_chunk_records)
    return vector_store


def make_chunk(index: int, document_id: str = "bulk-document", **overrides):
    values = {
        "chunk_id": f"{document_id}-p{index + 1}-c0",
        "document_id": document_id,
        "document_title": "Toplu Test Belgesi",
        "source_url": "https://www.example.edu.tr/toplu",
        "page_number": index + 1,
        "section_title": None,
        "text": (
            f"MADDE {index + 1} – Öğrenci sınav, ders kaydı, staj ve mezuniyet "
            f"işlemleri için {index + 1}. maddedeki kurallara uyar."
        ),
    }
    values.update(overrides)
    return ChunkRecord(**values)


def search(store: QdrantStore, method: str, question: str, limit: int = 20):
    return {
        "dense": store.search_dense,
        "bm25": store.search_bm25,
        "hybrid": store.search_hybrid_rrf,
    }[method](question, limit)


# Embedder with the real models ---------------------------------------------------


def test_encode_query_is_384_dimensional_unit_vector(
    real_embedder: TextEmbedder,
) -> None:
    vector = real_embedder.encode_query("Bütünleme sınavına kimler girebilir?")

    assert real_embedder.dense_dimension == 384
    assert len(vector) == 384
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-3)


def test_turkish_stopwords_are_removed_on_every_platform(
    real_embedder: TextEmbedder,
) -> None:
    assert real_embedder.encode_sparse_query("için çok nasıl ve ile").indices == []
    assert real_embedder.encode_sparse_query("icin cok nasil").indices == []


@pytest.mark.parametrize(
    ("document_text", "query"),
    [
        ("İZİN BELGESİ", "izin belgesi"),
        ("öğrenci işleri", "OGRENCI ISLERI"),
        ("IELTS sınavı", "ielts"),
        ("serti\ufb01ka", "sertifika"),
        ("öğ\u00adrenci", "öğrenci"),
    ],
)
def test_bm25_terms_match_across_spelling_variants(
    real_embedder: TextEmbedder,
    document_text: str,
    query: str,
) -> None:
    document_terms = set(real_embedder.encode_sparse([document_text])[0].indices)
    query_terms = set(real_embedder.encode_sparse_query(query).indices)

    assert query_terms and query_terms <= document_terms


# Collection schema ---------------------------------------------------------------


def test_collection_schema_matches_contract(vector_store: QdrantStore) -> None:
    vector_store.create_collection()

    config = vector_store.client.get_collection(vector_store.collection_name).config
    dense = config.params.vectors[DENSE_VECTOR_NAME]
    sparse = config.params.sparse_vectors[SPARSE_VECTOR_NAME]

    assert dense.size == 384
    assert dense.distance == models.Distance.COSINE
    assert sparse.modifier == models.Modifier.IDF
    assert config.metadata["dense_model"] == "intfloat/multilingual-e5-small"
    assert config.metadata["sparse_model"] == "Qdrant/bm25"


def test_create_collection_is_idempotent(
    indexed_vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    indexed_vector_store.create_collection()

    assert indexed_vector_store.count_points() == len(sample_chunk_records)


def test_recreate_drops_existing_points(indexed_vector_store: QdrantStore) -> None:
    indexed_vector_store.create_collection(recreate=True)

    assert indexed_vector_store.count_points() == 0


def test_wrong_dimension_collection_is_not_changed_silently(
    make_store,
    qdrant_test_client: QdrantClient,
) -> None:
    store = make_store()
    qdrant_test_client.create_collection(
        store.collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )

    with pytest.raises(CollectionSchemaError, match="dense size is 128"):
        store.create_collection()

    with pytest.raises(CollectionSchemaError):
        store.search_dense("Staj süresi nedir?")

    with pytest.raises(CollectionSchemaError):
        store.replace_document_chunks([make_chunk(0)])

    params = qdrant_test_client.get_collection(store.collection_name).config.params
    assert params.vectors[DENSE_VECTOR_NAME].size == 128


def test_unnamed_or_missing_vectors_are_rejected(
    make_store,
    qdrant_test_client: QdrantClient,
) -> None:
    store = make_store()
    qdrant_test_client.create_collection(
        store.collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    with pytest.raises(CollectionSchemaError) as error:
        store.create_collection()

    assert "'dense' is missing" in str(error.value)
    assert "'bm25' is missing" in str(error.value)


@pytest.mark.skipif(
    not os.environ.get("QDRANT_TEST_URL"),
    reason="needs a Qdrant server (set QDRANT_TEST_URL)",
)
def test_concurrent_first_upserts_from_separate_clients_do_not_fail(
    real_embedder: TextEmbedder,
) -> None:
    """Several processes indexing into a fresh collection at the same moment."""
    url = os.environ["QDRANT_TEST_URL"]
    name = f"test_race_{uuid.uuid4().hex[:12]}"
    clients = [create_qdrant_client(url) for _ in range(4)]
    stores = [
        QdrantStore(client=client, embedder=real_embedder, collection_name=name)
        for client in clients
    ]
    errors = []
    barrier = threading.Barrier(len(stores))

    def worker(index: int, store: QdrantStore) -> None:
        barrier.wait()
        try:
            store.upsert_chunks([make_chunk(index)])
        except Exception as exc:  # noqa: BLE001 - collect for the assert
            errors.append(exc)

    try:
        threads = [
            threading.Thread(target=worker, args=(index, store))
            for index, store in enumerate(stores)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert stores[0].count_points() == len(stores)
    finally:
        if clients[0].collection_exists(name):
            clients[0].delete_collection(name)
        for client in clients:
            client.close()


# Indexing -------------------------------------------------------------------------


def test_upsert_uses_deterministic_uuid_and_keeps_chunk_id(
    vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    assert vector_store.upsert_chunks(sample_chunk_records) is None

    chunk = sample_chunk_records[0]
    points = vector_store.client.retrieve(
        vector_store.collection_name,
        ids=[chunk_point_id(chunk.chunk_id)],
        with_payload=True,
    )

    assert len(points) == 1
    assert str(points[0].id) == chunk_point_id(chunk.chunk_id)
    assert points[0].payload["chunk_id"] == chunk.chunk_id


def test_payload_contains_contract_fields(
    indexed_vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    chunk = next(item for item in sample_chunk_records if item.section_title is None)
    point = indexed_vector_store.client.retrieve(
        indexed_vector_store.collection_name,
        ids=[chunk_point_id(chunk.chunk_id)],
        with_payload=True,
    )[0]

    assert set(point.payload) == set(PAYLOAD_FIELDS)
    assert point.payload["text"] == chunk.text
    assert point.payload["section_title"] is None
    assert point.payload["language"] == "tr"
    assert point.payload["content_hash"] == compute_content_hash(chunk.text)


def test_same_chunk_twice_does_not_increase_point_count(
    indexed_vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    changed = sample_chunk_records[0].model_copy(
        update={"text": "Güncellenmiş devam zorunluluğu metni."},
    )

    indexed_vector_store.upsert_chunks(sample_chunk_records)
    indexed_vector_store.upsert_chunks([changed])

    assert indexed_vector_store.count_points() == len(sample_chunk_records)

    point = indexed_vector_store.client.retrieve(
        indexed_vector_store.collection_name,
        ids=[chunk_point_id(changed.chunk_id)],
        with_payload=True,
    )[0]
    assert point.payload["text"] == "Güncellenmiş devam zorunluluğu metni."


def test_stopword_only_chunk_is_indexed_for_dense_search(
    vector_store: QdrantStore,
) -> None:
    vector_store.upsert_chunks([make_chunk(0, text="ve ile için çok")])

    assert vector_store.count_points() == 1
    assert vector_store.search_dense("ve ile", 5)


def test_upsert_batches_larger_than_batch_size(make_store) -> None:
    store = make_store(batch_size=4)

    store.upsert_chunks([make_chunk(index) for index in range(10)])

    assert store.count_points() == 10


# Deleting -------------------------------------------------------------------------


def test_delete_document_only_removes_that_document(
    indexed_vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    regulation_count = sum(
        chunk.document_id == REGULATION_DOC for chunk in sample_chunk_records
    )

    assert indexed_vector_store.delete_document(REGULATION_DOC) is None
    assert indexed_vector_store.count_points(REGULATION_DOC) == 0
    assert indexed_vector_store.count_points() == (
        len(sample_chunk_records) - regulation_count
    )
    assert indexed_vector_store.count_points(INTERNSHIP_DOC) == 2

    results = indexed_vector_store.search_dense("Bütünleme sınavı", 20)
    assert all(chunk.document_id != REGULATION_DOC for chunk in results)


def test_deleting_unknown_or_repeated_documents_is_a_no_op(
    indexed_vector_store: QdrantStore,
) -> None:
    indexed_vector_store.delete_document("unknown-document")
    indexed_vector_store.delete_document(ERASMUS_DOC)
    indexed_vector_store.delete_document(ERASMUS_DOC)

    assert indexed_vector_store.count_points(ERASMUS_DOC) == 0


def test_delete_document_accepts_uuid_objects(
    indexed_vector_store: QdrantStore,
) -> None:
    indexed_vector_store.delete_document(uuid.UUID(ERASMUS_DOC))

    assert indexed_vector_store.count_points(ERASMUS_DOC) == 0


def test_replace_document_chunks_removes_stale_chunks(
    indexed_vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    regulation = [
        chunk for chunk in sample_chunk_records if chunk.document_id == REGULATION_DOC
    ]

    indexed_vector_store.replace_document_chunks(regulation[:2])

    assert indexed_vector_store.count_points(REGULATION_DOC) == 2
    assert indexed_vector_store.count_points(INTERNSHIP_DOC) == 2


# Searching -------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_search_never_returns_more_than_limit(
    vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
    method: str,
) -> None:
    vector_store.upsert_chunks(
        sample_chunk_records + [make_chunk(index) for index in range(25)],
    )
    question = "Öğrenci sınav ve ders kaydı kuralları nelerdir?"

    assert len(search(vector_store, method, question, 20)) == 20
    assert len(search(vector_store, method, question, 3)) == 3


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_invalid_limit_is_rejected(indexed_vector_store: QdrantStore, limit) -> None:
    with pytest.raises((TypeError, ValueError)):
        indexed_vector_store.search_dense("Staj süresi", limit)


@pytest.mark.parametrize(("question", "expected_chunk_id"), RELEVANCE_CASES)
def test_relevant_regulation_chunk_is_in_top_5(
    indexed_vector_store: QdrantStore,
    question: str,
    expected_chunk_id: str,
) -> None:
    top_5 = {
        method: [
            chunk.chunk_id
            for chunk in search(indexed_vector_store, method, question, 5)
        ]
        for method in ("dense", "bm25", "hybrid")
    }

    assert any(expected_chunk_id in ids for ids in top_5.values()), top_5
    assert expected_chunk_id in top_5["hybrid"], top_5


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("dense", RetrievalMethod.DENSE),
        ("bm25", RetrievalMethod.BM25),
        ("hybrid", RetrievalMethod.HYBRID),
    ],
)
def test_results_are_retrieved_chunks_with_source_metadata(
    indexed_vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
    method: str,
    expected: RetrievalMethod,
) -> None:
    by_id = {chunk.chunk_id: chunk for chunk in sample_chunk_records}
    results = search(indexed_vector_store, method, "Staj başvurusu ve sigorta", 5)

    assert results
    scores = [chunk.retrieval_score for chunk in results]
    assert scores == sorted(scores, reverse=True)

    for result in results:
        assert type(result) is RetrievedChunk
        assert result.retrieval_method == expected
        assert result.rerank_score is None
        original = by_id[result.chunk_id]
        assert result.model_dump(include=set(ChunkRecord.model_fields)) == (
            original.model_dump()
        )


def test_bm25_query_with_only_stopwords_returns_empty(
    indexed_vector_store: QdrantStore,
) -> None:
    assert indexed_vector_store.search_bm25("için ve ile", 20) == []
    assert indexed_vector_store.search_hybrid_rrf("için ve ile", 5)


def test_bm25_finds_uppercase_turkish_text(vector_store: QdrantStore) -> None:
    vector_store.upsert_chunks(
        [
            make_chunk(0, text="İZİN VE İLİŞİK KESME İŞLEMLERİ"),
            make_chunk(1, text="Kütüphane çalışma saatleri"),
        ],
    )

    results = vector_store.search_bm25("ilisik kesme", 5)

    assert [chunk.page_number for chunk in results] == [1]


def test_searching_before_collection_exists_raises_clear_error(
    vector_store: QdrantStore,
) -> None:
    with pytest.raises(CollectionNotFoundError) as error:
        vector_store.search_dense("Staj süresi nedir?")

    assert error.value.status_code == 503
    vector_store.delete_document("any")
    assert vector_store.count_points() == 0
    assert vector_store.health_status() == "collection_missing"


# Errors and adapters ----------------------------------------------------------------


def test_unreachable_qdrant_raises_controlled_error(
    real_embedder: TextEmbedder,
) -> None:
    client = create_qdrant_client("http://127.0.0.1:9")
    store = QdrantStore(client=client, embedder=real_embedder, collection_name="down")

    with pytest.raises(VectorStoreUnavailableError) as error:
        store.search_dense("Staj süresi nedir?")

    assert error.value.status_code == 503
    assert error.value.code == "service_unavailable"
    assert "Qdrant" in error.value.message

    with pytest.raises(VectorStoreUnavailableError):
        store.create_collection()

    assert store.health_status() == "unavailable"
    client.close()


def test_async_adapter_round_trip(
    indexed_vector_store: QdrantStore,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    vector_store = QdrantVectorStore(indexed_vector_store)

    async def scenario():
        searches = await asyncio.gather(
            vector_store.search_dense("Staj kaç gün?", 5),
            vector_store.search_bm25("Staj kaç gün?", 5),
            vector_store.search_hybrid_rrf("Staj kaç gün?", 5),
        )
        await vector_store.delete_document(INTERNSHIP_DOC)
        after_delete = indexed_vector_store.count_points(INTERNSHIP_DOC)
        await vector_store.upsert_chunks(sample_chunk_records)
        return searches, after_delete, await vector_store.health_status()

    searches, after_delete, health = asyncio.run(scenario())

    assert [results[0].retrieval_method for results in searches] == [
        RetrievalMethod.DENSE,
        RetrievalMethod.BM25,
        RetrievalMethod.HYBRID,
    ]
    assert after_delete == 0
    assert indexed_vector_store.count_points() == len(sample_chunk_records)
    assert health == "ok"


def test_local_folder_mode_persists_and_handles_threads(
    tmp_path,
    real_embedder: TextEmbedder,
    sample_chunk_records: list[ChunkRecord],
) -> None:
    location = str(tmp_path / "qdrant")
    store = QdrantStore(
        client=create_qdrant_client(location),
        embedder=real_embedder,
        collection_name="local_test",
    )
    store.upsert_chunks(sample_chunk_records)
    vector_store = QdrantVectorStore(store)

    async def many_searches():
        return await asyncio.gather(
            *(vector_store.search_hybrid_rrf("Staj süresi", 3) for _ in range(6)),
        )

    assert all(len(results) == 3 for results in asyncio.run(many_searches()))
    store.close()

    reopened = QdrantStore(
        client=create_qdrant_client(location),
        embedder=real_embedder,
        collection_name="local_test",
    )
    assert reopened.count_points() == len(sample_chunk_records)
    reopened.close()
