"""Offline tests for QdrantStore: fake embedder + in-memory Qdrant, no models."""

import asyncio
import inspect
import json
import threading
import typing
import uuid
import zlib

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from backend.app.core.config import Settings
from backend.app.core.exceptions import ApplicationError
from backend.app.main import create_app
from backend.app.ports.vector_store import VectorStore
from backend.app.rag import qdrant_store as qdrant_store_module
from backend.app.rag.embedder import EmbeddingModelError
from backend.app.rag.qdrant_store import (
    ChunkIdConflictError,
    CollectionNotFoundError,
    CollectionSchemaError,
    QdrantStore,
    QdrantVectorStore,
    VectorStoreConfigError,
    VectorStoreError,
    VectorStoreUnavailableError,
    _validate_limit,
    chunk_point_id,
    compute_content_hash,
    create_qdrant_client,
    get_vector_store,
)
from backend.app.schemas.retrieval import ChunkRecord
from backend.app.services.document_processor import get_document_processor

DIMENSION = 8


class FakeEmbedder:
    """Deterministic stand-in for TextEmbedder (hash-based vectors)."""

    def __init__(self, dense_model_name: str = "fake-dense") -> None:
        self.dense_model_name = dense_model_name
        self.sparse_model_name = "fake-sparse"
        self.fail_documents = False

    @property
    def dense_dimension(self) -> int:
        return DIMENSION

    def load(self) -> None:
        pass

    def _vector(self, text: str) -> list[float]:
        rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
        vector = rng.normal(size=DIMENSION)
        return (vector / np.linalg.norm(vector)).tolist()

    def encode_documents(self, texts):
        if self.fail_documents:
            raise EmbeddingModelError("simulated model failure")
        return [self._vector(text) for text in texts]

    def encode_query(self, question):
        return self._vector(question)

    def _sparse(self, text: str) -> models.SparseVector:
        terms = sorted({zlib.crc32(word.encode()) for word in text.lower().split()})
        return models.SparseVector(indices=terms, values=[1.0] * len(terms))

    def encode_sparse(self, texts):
        return [self._sparse(text) for text in texts]

    def encode_sparse_query(self, question):
        return self._sparse(question)


def chunk(document_id: str, index: int, text: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{document_id}-p{index + 1}-c0",
        document_id=document_id,
        document_title="Test Yönetmeliği",
        source_url="https://www.example.edu.tr/test",
        page_number=index + 1,
        section_title=None,
        text=text or f"belge {document_id} madde {index} staj sınav",
    )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def memory_store(fake_embedder: FakeEmbedder):
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, embedder=fake_embedder, collection_name="unit")
    yield store
    client.close()


def stored_texts(store: QdrantStore, document_id: str) -> list[str]:
    points, _ = store.client.scroll(
        store.collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
            ],
        ),
        limit=100,
        with_payload=True,
    )
    return sorted(point.payload["text"] for point in points)


# Point IDs and payload -------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_id",
    [
        "3f1c2a9e-8b7d-4c21-9f3e-1a2b3c4d5e6f-p4-c2",
        "Yönetmelik / sayfa 4 / parça 2",
        "x" * 2000,
    ],
)
def test_point_id_is_a_deterministic_uuid_for_any_chunk_id(chunk_id: str) -> None:
    point_id = chunk_point_id(chunk_id)

    assert uuid.UUID(point_id).version == 5
    assert chunk_point_id(chunk_id) == point_id
    assert chunk_point_id(chunk_id + "x") != point_id


def test_text_chunk_id_is_kept_in_payload_and_uuid_used_as_point_id(
    memory_store: QdrantStore,
) -> None:
    record = chunk("doc-a", 0)

    memory_store.upsert_chunks([record])

    points, _ = memory_store.client.scroll("unit", with_payload=True)
    assert len(points) == 1
    assert str(points[0].id) == chunk_point_id(record.chunk_id)
    assert points[0].payload["chunk_id"] == record.chunk_id
    assert points[0].payload["content_hash"] == compute_content_hash(record.text)


def test_content_hash_ignores_whitespace_and_unicode_form() -> None:
    assert compute_content_hash("öğrenci  işleri\n") == compute_content_hash(
        "o\u0308g\u0306renci işleri",
    )


def test_uuid_document_id_is_stored_and_deleted_as_the_same_string(
    memory_store: QdrantStore,
) -> None:
    document_uuid = uuid.uuid4()
    memory_store.upsert_chunks([chunk(str(document_uuid), 0)])

    memory_store.delete_document(document_uuid)  # UUID object from DocumentResponse

    assert memory_store.count_points() == 0


# Port contract ----------------------------------------------------------------------


def test_async_adapter_matches_vector_store_port() -> None:
    for name, port_method in inspect.getmembers(VectorStore, inspect.isfunction):
        if name.startswith("_"):
            continue

        implementation = getattr(QdrantVectorStore, name)
        port_hints = typing.get_type_hints(port_method)
        implementation_hints = typing.get_type_hints(implementation)

        assert inspect.iscoroutinefunction(implementation), name
        assert implementation_hints == port_hints, name
        assert list(inspect.signature(implementation).parameters) == list(
            inspect.signature(port_method).parameters,
        ), name


def test_upsert_and_delete_return_none(memory_store: QdrantStore) -> None:
    vector_store = QdrantVectorStore(memory_store)

    async def scenario():
        return (
            await vector_store.upsert_chunks([chunk("doc-a", 0)]),
            await vector_store.delete_document("doc-a"),
        )

    assert asyncio.run(scenario()) == (None, None)


# Indexing safety ----------------------------------------------------------------------


def test_replace_keeps_old_version_when_embedding_fails(
    memory_store: QdrantStore,
    fake_embedder: FakeEmbedder,
) -> None:
    memory_store.upsert_chunks([chunk("doc-a", i, f"eski {i}") for i in range(3)])
    fake_embedder.fail_documents = True

    with pytest.raises(EmbeddingModelError):
        memory_store.replace_document_chunks([chunk("doc-a", 0, "yeni 0")])

    assert stored_texts(memory_store, "doc-a") == ["eski 0", "eski 1", "eski 2"]


def test_replace_writes_new_version_and_removes_stale_chunks(
    memory_store: QdrantStore,
) -> None:
    memory_store.upsert_chunks([chunk("doc-a", i, f"eski {i}") for i in range(3)])
    memory_store.upsert_chunks([chunk("doc-b", 0, "başka belge")])

    memory_store.replace_document_chunks([chunk("doc-a", 0, "yeni 0")])

    assert stored_texts(memory_store, "doc-a") == ["yeni 0"]
    assert stored_texts(memory_store, "doc-b") == ["başka belge"]


def test_concurrent_replaces_of_one_document_do_not_mix_versions(
    memory_store: QdrantStore,
) -> None:
    versions = [
        [chunk("doc-a", i, f"sürüm bir {i}") for i in range(4)],
        [chunk("doc-a", i, f"sürüm iki {i}") for i in range(2)],
    ]
    barrier = threading.Barrier(2)

    def replace(records):
        barrier.wait()
        memory_store.replace_document_chunks(records)

    threads = [threading.Thread(target=replace, args=(v,)) for v in versions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    texts = stored_texts(memory_store, "doc-a")
    assert texts in (
        [f"sürüm bir {i}" for i in range(4)],
        [f"sürüm iki {i}" for i in range(2)],
    )


def test_same_chunk_id_twice_updates_instead_of_duplicating(
    memory_store: QdrantStore,
) -> None:
    memory_store.upsert_chunks([chunk("doc-a", 0, "ilk")])
    memory_store.upsert_chunks([chunk("doc-a", 0, "güncel")])

    assert stored_texts(memory_store, "doc-a") == ["güncel"]


def test_chunk_id_cannot_move_to_another_document(memory_store: QdrantStore) -> None:
    memory_store.upsert_chunks([chunk("doc-a", 0)])
    stolen = chunk("doc-a", 0).model_copy(update={"document_id": "doc-b"})

    with pytest.raises(ChunkIdConflictError) as error:
        memory_store.replace_document_chunks([stolen])

    assert error.value.status_code == 409
    assert memory_store.count_points("doc-a") == 1


def test_chunk_id_conflict_inside_one_call_is_rejected(
    memory_store: QdrantStore,
) -> None:
    first = chunk("doc-a", 0)
    second = first.model_copy(update={"document_id": "doc-b"})

    with pytest.raises(ChunkIdConflictError):
        memory_store.upsert_chunks([first, second])


@pytest.mark.parametrize("field", ["chunk_id", "document_id"])
def test_blank_ids_are_rejected_before_indexing(
    memory_store: QdrantStore,
    field: str,
) -> None:
    record = chunk("doc-a", 0).model_copy(update={field: "  "})

    with pytest.raises(ValueError):
        memory_store.upsert_chunks([record])

    with pytest.raises(ValueError):
        memory_store.delete_document("  ")


def test_collection_deleted_externally_is_recreated_on_next_upsert(
    memory_store: QdrantStore,
) -> None:
    memory_store.upsert_chunks([chunk("doc-a", 0)])
    memory_store.client.delete_collection("unit")

    with pytest.raises(CollectionNotFoundError):
        memory_store.search_dense("staj")

    memory_store.upsert_chunks([chunk("doc-a", 1)])
    assert memory_store.count_points() == 1


# Collection schema ---------------------------------------------------------------------


def test_collection_built_with_another_model_is_rejected(
    memory_store: QdrantStore,
) -> None:
    memory_store.upsert_chunks([chunk("doc-a", 0)])
    other = QdrantStore(
        client=memory_store.client,
        embedder=FakeEmbedder(dense_model_name="another-384-model"),
        collection_name="unit",
    )

    with pytest.raises(CollectionSchemaError, match="dense_model") as error:
        other.create_collection()

    assert error.value.status_code == 503
    assert memory_store.count_points() == 1


def test_concurrent_create_collection_in_one_process_is_safe(
    memory_store: QdrantStore,
) -> None:
    errors = []
    barrier = threading.Barrier(6)

    def create():
        barrier.wait()
        try:
            memory_store.create_collection()
        except Exception as exc:  # noqa: BLE001 - collect for the assert
            errors.append(exc)

    threads = [threading.Thread(target=create) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_create_tolerates_collection_created_by_another_process(
    memory_store: QdrantStore,
    monkeypatch,
) -> None:
    real_create = memory_store.client.create_collection

    def racing_create(**kwargs):
        real_create(**kwargs)  # the other process wins the race ...
        raise _unexpected(500, b'{"status":{"error":"Service internal error"}}')

    monkeypatch.setattr(memory_store.client, "create_collection", racing_create)

    memory_store.create_collection()

    assert memory_store.health_status() == "ok"


def test_create_waits_while_concurrent_collection_is_still_initializing(
    memory_store: QdrantStore,
    monkeypatch,
) -> None:
    real_create = memory_store.client.create_collection
    real_get = memory_store.client.get_collection
    failures = {"left": 2}

    def racing_create(**kwargs):
        real_create(**kwargs)
        raise _unexpected(500)

    def initializing_get(*args, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise _unexpected(500, b'{"status":{"error":"0 of 0 read operations"}}')
        return real_get(*args, **kwargs)

    monkeypatch.setattr(memory_store.client, "create_collection", racing_create)
    monkeypatch.setattr(memory_store.client, "get_collection", initializing_get)
    monkeypatch.setattr(qdrant_store_module, "CONCURRENT_CREATE_POLL_SECONDS", 0)

    memory_store.create_collection()

    assert failures["left"] == 0
    assert memory_store.health_status() == "ok"


def test_failed_create_without_concurrent_creator_raises(
    memory_store: QdrantStore,
    monkeypatch,
) -> None:
    def failing_create(**kwargs):
        raise _unexpected(500)

    monkeypatch.setattr(memory_store.client, "create_collection", failing_create)
    monkeypatch.setattr(qdrant_store_module, "CONCURRENT_CREATE_POLL_SECONDS", 0)

    with pytest.raises(VectorStoreUnavailableError):
        memory_store.create_collection()


# Search guards --------------------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\u200b\u200b\u200b"])
def test_invisible_question_returns_no_results(
    memory_store: QdrantStore,
    question: str,
) -> None:
    memory_store.upsert_chunks([chunk("doc-a", 0)])

    assert memory_store.search_dense(question) == []
    assert memory_store.search_bm25(question) == []
    assert memory_store.search_hybrid_rrf(question) == []


@pytest.mark.parametrize("limit", [1, 20, np.int64(5)])
def test_valid_limits(limit) -> None:
    assert _validate_limit(limit) == int(limit)


@pytest.mark.parametrize("limit", [0, -1])
def test_limits_below_one_are_rejected(limit) -> None:
    with pytest.raises(ValueError):
        _validate_limit(limit)


@pytest.mark.parametrize("limit", [True, 2.5, "3", None])
def test_non_integer_limits_are_rejected(limit) -> None:
    with pytest.raises(TypeError):
        _validate_limit(limit)


# Error mapping ----------------------------------------------------------------------------


def _unexpected(status: int, content: bytes = b"{}") -> UnexpectedResponse:
    return UnexpectedResponse(status, "", content, httpx.Headers())


def _validation_error() -> ValidationError:
    try:
        models.VectorParams(size="not a number", distance=models.Distance.COSINE)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (_unexpected(404), CollectionNotFoundError),
        (_unexpected(400), VectorStoreError),
        (_unexpected(401), VectorStoreError),
        (_unexpected(429), VectorStoreUnavailableError),
        (_unexpected(500), VectorStoreUnavailableError),
        (_unexpected(503), VectorStoreUnavailableError),
        (ResourceExhaustedResponse("slow down", 1), VectorStoreUnavailableError),
        (
            ResponseHandlingException(httpx.ConnectError("refused")),
            VectorStoreUnavailableError,
        ),
        (ResponseHandlingException(_validation_error()), VectorStoreError),
        (json.JSONDecodeError("not json", "<html>", 0), VectorStoreUnavailableError),
        (httpx.ReadTimeout("timeout"), VectorStoreUnavailableError),
        (ValueError("Collection unit not found"), CollectionNotFoundError),
        (ValueError("bad input"), VectorStoreError),
        (RuntimeError("client closed"), VectorStoreError),
        (AssertionError(), VectorStoreError),
    ],
)
def test_client_errors_become_application_errors(
    memory_store: QdrantStore,
    raised: Exception,
    expected: type,
) -> None:
    with pytest.raises(expected) as error, memory_store._qdrant_errors("test"):
        raise raised

    assert type(error.value) is expected
    assert isinstance(error.value, ApplicationError)
    assert error.value.message  # Turkish user-facing message


def test_unexpected_404_on_exists_check_means_wrong_url(
    memory_store: QdrantStore,
) -> None:
    errors = memory_store._qdrant_errors("check", not_found_is_missing=False)

    with pytest.raises(VectorStoreUnavailableError), errors:
        raise _unexpected(404)


def test_health_status_never_raises(memory_store: QdrantStore, monkeypatch) -> None:
    assert memory_store.health_status() == "collection_missing"
    memory_store.create_collection()
    assert memory_store.health_status() == "ok"

    def broken(*args, **kwargs):
        raise RuntimeError("anything at all")

    monkeypatch.setattr(memory_store.client, "collection_exists", broken)
    assert memory_store.health_status() == "unavailable"


# QDRANT_URL handling -----------------------------------------------------------------------


def test_server_urls_are_accepted_case_insensitively() -> None:
    client = create_qdrant_client("HTTP://127.0.0.1:6333")

    assert type(client._client).__name__ == "QdrantRemote"
    client.close()


@pytest.mark.parametrize("value", ["localhost:6333", "qdrant:6333", "qdrant", ""])
def test_ambiguous_qdrant_url_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="QDRANT_URL") as caught:
        create_qdrant_client(value)

    # Still a ValueError for scripts, but a clear 503 for the API.
    assert isinstance(caught.value, VectorStoreConfigError)
    assert caught.value.status_code == 503
    assert caught.value.code == "vector_store_misconfigured"
    assert "QDRANT_URL" in caught.value.message


def test_local_folder_is_locked_for_a_second_client(tmp_path) -> None:
    folder = str(tmp_path / "qdrant")
    first = create_qdrant_client(folder)

    try:
        with pytest.raises(VectorStoreUnavailableError):
            create_qdrant_client(folder)
    finally:
        first.close()


# Lazy engine: a wrong QDRANT_URL must not break the FastAPI dependency -----------------


def test_async_store_creates_the_engine_once_on_first_use(
    fake_embedder: FakeEmbedder,
) -> None:
    created: list[QdrantStore] = []

    def factory() -> QdrantStore:
        store = QdrantStore(
            client=QdrantClient(":memory:"),
            embedder=fake_embedder,
            collection_name="lazy",
        )
        created.append(store)
        return store

    vector_store = QdrantVectorStore(store_factory=factory)
    assert created == []

    async def check_many() -> list[str]:
        return await asyncio.gather(*(vector_store.health_status() for _ in range(8)))

    assert asyncio.run(check_many()) == ["collection_missing"] * 8
    assert len(created) == 1
    assert vector_store.store is created[0]
    created[0].close()


def test_bad_qdrant_url_fails_the_operation_and_health_reports_it(
    fake_embedder: FakeEmbedder,
) -> None:
    attempts = []

    def factory() -> QdrantStore:
        attempts.append(1)
        return QdrantStore(
            embedder=fake_embedder,
            settings=Settings(qdrant_url="localhost:6333"),
        )

    vector_store = QdrantVectorStore(store_factory=factory)

    assert asyncio.run(vector_store.health_status()) == "unavailable"

    with pytest.raises(VectorStoreConfigError):
        asyncio.run(vector_store.search_hybrid_rrf("staj süresi"))

    # A failed creation is not cached, so a fixed setting is picked up later.
    assert len(attempts) == 2


def test_locked_local_folder_is_reported_as_unavailable(
    tmp_path,
    fake_embedder: FakeEmbedder,
) -> None:
    folder = str(tmp_path / "qdrant")
    holder = create_qdrant_client(folder)  # e.g. index_documents.py still running

    try:
        vector_store = QdrantVectorStore(
            store_factory=lambda: QdrantStore(
                embedder=fake_embedder,
                settings=Settings(qdrant_url=folder),
            ),
        )

        assert asyncio.run(vector_store.health_status()) == "unavailable"

        with pytest.raises(VectorStoreUnavailableError):
            asyncio.run(vector_store.upsert_chunks([chunk("doc-a", 0)]))
    finally:
        holder.close()


class _OneChunkProcessor:
    def process_document(self, file_path, document) -> list[ChunkRecord]:
        return [chunk(str(document.document_id), 0)]


def _tiny_pdf() -> bytes:
    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "MADDE 1 - Test")
    content = document.tobytes()
    document.close()
    return content


def test_api_with_bad_qdrant_url_reports_degraded_and_clear_503(
    tmp_path,
    monkeypatch,
) -> None:
    # The real get_vector_store() dependency, built from a wrong QDRANT_URL.
    monkeypatch.setattr(qdrant_store_module, "_default_store", None)
    monkeypatch.setattr(qdrant_store_module, "_default_vector_store", None)
    monkeypatch.setattr(
        qdrant_store_module,
        "get_settings",
        lambda: Settings(qdrant_url="localhost:6333"),
    )
    monkeypatch.chdir(tmp_path)  # uploads are stored under ./data/documents

    app = create_app(
        Settings(
            app_env="testing",
            database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        ),
    )
    app.dependency_overrides[get_document_processor] = _OneChunkProcessor

    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/api/health")
        upload = client.post(
            "/api/documents/upload",
            data={
                "title": "Test Yönetmeliği",
                "source_url": "https://example.edu.tr/t.pdf",
            },
            files={"file": ("test.pdf", _tiny_pdf(), "application/pdf")},
        )
        documents = client.get("/api/documents").json()["documents"]

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["qdrant"] == "unavailable"
    assert health.json()["database"] == "ok"

    assert upload.status_code == 503
    assert upload.json()["error"] == "vector_store_misconfigured"
    assert [document["status"] for document in documents] == ["failed"]

    assert isinstance(get_vector_store(), QdrantVectorStore)


@pytest.mark.parametrize("value", ["http://localhost:63a3", "https://[::1"])
def test_malformed_server_url_is_a_config_error(value: str) -> None:
    with pytest.raises(VectorStoreConfigError):
        create_qdrant_client(value)


def test_existing_file_as_local_folder_is_a_config_error(tmp_path) -> None:
    not_a_folder = tmp_path / "qdrant.txt"
    not_a_folder.write_text("x", encoding="utf-8")

    with pytest.raises(VectorStoreConfigError):
        create_qdrant_client(str(not_a_folder))


def test_close_never_creates_the_engine(fake_embedder: FakeEmbedder) -> None:
    attempts = []

    def factory() -> QdrantStore:
        attempts.append(1)
        raise VectorStoreConfigError("QDRANT_URL 'x' is not usable")

    vector_store = QdrantVectorStore(store_factory=factory)
    vector_store.close()  # e.g. in a script's finally block after a failure
    assert attempts == []

    store = QdrantStore(
        client=QdrantClient(":memory:"),
        embedder=fake_embedder,
        collection_name="close",
    )
    created = QdrantVectorStore(store)
    created.close()

    with pytest.raises(RuntimeError, match="closed"):
        store.client.get_collections()


def test_async_store_calls_the_engines_own_methods(memory_store: QdrantStore) -> None:
    class CountingStore(QdrantStore):
        searches = 0

        def search_dense(self, question, limit=20):
            CountingStore.searches += 1
            return []

    store = CountingStore(
        client=memory_store.client,
        embedder=memory_store._embedder,
        collection_name="counting",
    )

    assert asyncio.run(QdrantVectorStore(store).search_dense("staj")) == []
    assert CountingStore.searches == 1
