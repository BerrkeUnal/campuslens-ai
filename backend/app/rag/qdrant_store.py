"""Qdrant storage and the three retrieval methods: dense, BM25 and hybrid RRF.

API code uses the async implementation of
``backend.app.ports.vector_store.VectorStore``::

    vector_store = get_vector_store()
    chunks = await vector_store.search_hybrid_rrf(question, 20)

``QdrantStore`` is the synchronous engine behind it. Use it directly only in
scripts and offline evaluation: calling it from async code would block the
event loop. Callers only ever receive the shared ``RetrievedChunk`` schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import operator
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any

import httpx
from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from qdrant_client.common.client_exceptions import (
    QdrantException,
    ResourceExhaustedResponse,
)
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import ApplicationError, ServiceUnavailableError
from backend.app.rag.embedder import (
    BM25_PREPROCESSING_VERSION,
    TextEmbedder,
    get_embedder,
    has_searchable_text,
    normalize_text,
)
from backend.app.schemas.retrieval import (
    ChunkRecord,
    RetrievalMethod,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
DOCUMENT_LANGUAGE = "tr"
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_UPSERT_BATCH_SIZE = 64
QDRANT_TIMEOUT_SECONDS = 10
CONCURRENT_CREATE_POLLS = 20
CONCURRENT_CREATE_POLL_SECONDS = 0.25

# Fixed namespace so the same chunk_id always maps to the same point ID.
CHUNK_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "campuslens-ai/chunks")

PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "source_url",
    "page_number",
    "section_title",
    "text",
    "language",
    "content_hash",
)

_URL_SCHEME = re.compile(r"^(https?)://", re.IGNORECASE)
_HOST_PORT = re.compile(r"^[\w.-]+:\d+(/|$)")


class VectorStoreError(ApplicationError):
    """Qdrant rejected an operation or returned something unusable."""

    def __init__(
        self,
        detail: str,
        *,
        code: str = "vector_store_error",
        message: str = "Belge arama servisinde beklenmeyen bir hata oluştu.",
        status_code: int = 500,
    ) -> None:
        super().__init__(code=code, message=message, status_code=status_code)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


class CollectionSchemaError(VectorStoreError):
    """An existing collection does not match the CampusLens contract."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            detail,
            code="index_schema_mismatch",
            message=(
                "Belge arama indeksi beklenen yapıda değil. "
                "Lütfen sistem yöneticisine bildirin."
            ),
            status_code=503,
        )


class ChunkIdConflictError(VectorStoreError):
    """The same chunk_id was used for two different documents."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            detail,
            code="chunk_id_conflict",
            message="Aynı belge parçası kimliği birden fazla belgeye ait olamaz.",
            status_code=409,
        )


class VectorStoreUnavailableError(ServiceUnavailableError):
    """Qdrant cannot be reached or is temporarily overloaded (HTTP 503)."""

    def __init__(self, detail: str) -> None:
        super().__init__("Qdrant")
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


class CollectionNotFoundError(ApplicationError):
    """Searching before the collection has been created (HTTP 503)."""

    def __init__(self, collection_name: str) -> None:
        super().__init__(
            code="index_not_ready",
            message=(
                "Belge arama indeksi henüz oluşturulmamış. Lütfen önce belge yükleyin."
            ),
            status_code=503,
        )
        self.collection_name = collection_name

    def __str__(self) -> str:
        return f"Qdrant collection '{self.collection_name}' does not exist."


def chunk_point_id(chunk_id: str) -> str:
    """Deterministic UUID point ID for any chunk_id string.

    Qdrant only accepts UUIDs or unsigned integers as point IDs, so the
    original chunk_id is kept in the payload and the ID is derived from it.
    """
    return str(uuid.uuid5(CHUNK_POINT_NAMESPACE, chunk_id))


def compute_content_hash(text: str) -> str:
    """SHA-256 of the NFC-normalized, whitespace-collapsed chunk text."""
    normalized = " ".join(normalize_text(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_payload(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_title": chunk.document_title,
        "source_url": chunk.source_url,
        "page_number": chunk.page_number,
        "section_title": chunk.section_title,
        "text": chunk.text,
        "language": DOCUMENT_LANGUAGE,
        "content_hash": compute_content_hash(chunk.text),
    }


def create_qdrant_client(location: str) -> QdrantClient:
    """Create a client for a server URL, ``:memory:`` or a local folder.

    ``http(s)://host:port`` connects to a running Qdrant server. A folder path
    such as ``./data/qdrant_local`` uses Qdrant's embedded local mode, which
    needs no separate Qdrant process but can be opened by one process at a
    time. Anything else (e.g. ``localhost:6333`` without a scheme) is
    rejected instead of silently creating a folder.
    """
    location = location.strip()

    if location == ":memory:":
        return QdrantClient(location=":memory:")

    scheme = _URL_SCHEME.match(location)

    if scheme:
        url = scheme.group(1).lower() + location[scheme.end(1) :]
        return QdrantClient(
            url=url,
            timeout=QDRANT_TIMEOUT_SECONDS,
            check_compatibility=False,
        )

    if not _looks_like_folder(location):
        raise ValueError(
            f"QDRANT_URL '{location}' is not an http(s) URL, ':memory:' or a "
            "folder path. Use e.g. 'http://localhost:6333' for a Qdrant server "
            "or './data/qdrant_local' for embedded local mode.",
        )

    folder = os.path.abspath(location)
    logger.info("Using embedded local Qdrant storage at %s", folder)

    try:
        # Calls are serialized by QdrantStore's lock but may come from
        # different worker threads; macOS SQLite would reject that.
        return QdrantClient(path=folder, force_disable_check_same_thread=True)
    except RuntimeError as exc:
        raise VectorStoreUnavailableError(
            f"Local Qdrant folder {folder} is already used by another process: {exc}",
        ) from exc


def _looks_like_folder(location: str) -> bool:
    if not location or _HOST_PORT.match(location):
        return False

    return (
        location.startswith(".")
        or "/" in location
        or "\\" in location
        or os.path.isabs(location)
    )


def _is_local_client(client: QdrantClient) -> bool:
    try:
        from qdrant_client.local.qdrant_local import QdrantLocal
    except ImportError:  # pragma: no cover - depends on client internals
        return False

    return isinstance(getattr(client, "_client", None), QdrantLocal)


def _validate_limit(limit: int) -> int:
    """Accept any integer type (e.g. numpy ints) >= 1; reject bool/float/str."""
    if isinstance(limit, bool):
        raise TypeError("limit must be an integer, not bool.")

    try:
        value = operator.index(limit)
    except TypeError as exc:
        raise TypeError(
            f"limit must be an integer, not {type(limit).__name__}."
        ) from exc

    if value < 1:
        raise ValueError("limit must be at least 1.")

    return value


def _require_id(value: object, name: str) -> str:
    text = str(value)

    if not text.strip():
        raise ValueError(f"{name} cannot be empty.")

    return text


def _document_filter(
    document_id: str,
    exclude_point_ids: Sequence[str] = (),
) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
        ],
        must_not=(
            [models.HasIdCondition(has_id=list(exclude_point_ids))]
            if exclude_point_ids
            else None
        ),
    )


class QdrantStore:
    """Synchronous Qdrant engine for indexing and searching chunks."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: TextEmbedder | None = None,
        *,
        collection_name: str | None = None,
        settings: Settings | None = None,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._settings = settings or get_settings()
        self._client = client or create_qdrant_client(self._settings.qdrant_url)
        self._embedder = embedder or get_embedder()
        self.collection_name = collection_name or self._settings.qdrant_collection
        self.batch_size = batch_size

        self._collection_verified = False
        # Embedded local mode is not safe for concurrent use from threads.
        self._client_lock = (
            threading.RLock() if _is_local_client(self._client) else nullcontext()
        )
        self._create_lock = threading.Lock()
        self._document_locks: dict[str, threading.Lock] = {}
        self._document_locks_guard = threading.Lock()

    @property
    def client(self) -> QdrantClient:
        return self._client

    def close(self) -> None:
        with self._client_lock:
            self._client.close()

    # Collection management -------------------------------------------------

    def warm_up(self) -> None:
        """Load both models and create/validate the collection (app startup)."""
        self._embedder.load()
        self.create_collection()

    def create_collection(self, recreate: bool = False) -> None:
        """Create the collection, or validate it if it already exists.

        With ``recreate=True`` an existing collection and all its points are
        deleted first. A collection with a different schema is never changed
        silently; ``CollectionSchemaError`` is raised instead.
        """
        with self._create_lock:
            # Check Qdrant before loading the model so a wrong URL fails fast.
            exists = self.collection_exists()
            dimension = self._embedder.dense_dimension

            with self._qdrant_errors("create collection"):
                if exists and recreate:
                    logger.warning("Recreating collection %s", self.collection_name)
                    self._collection_verified = False
                    self._client.delete_collection(self.collection_name)
                    exists = False

                if exists:
                    payload_schema = self._validate_schema(dimension).payload_schema
                else:
                    payload_schema = self._create_new_collection(dimension)

                self._ensure_payload_index(payload_schema)

            self._collection_verified = True

    def collection_exists(self) -> bool:
        # A real server answers /exists with 200 even for a missing
        # collection, so a 404 here means a wrong URL, not a missing index.
        with self._qdrant_errors("check collection", not_found_is_missing=False):
            return self._client.collection_exists(self.collection_name)

    def count_points(self, document_id: str | None = None) -> int:
        """Count points in the collection, optionally for one document."""
        count_filter = (
            _document_filter(_require_id(document_id, "document_id"))
            if document_id is not None
            else None
        )

        try:
            with self._qdrant_errors("count points"):
                result = self._client.count(
                    collection_name=self.collection_name,
                    count_filter=count_filter,
                    exact=True,
                )
        except CollectionNotFoundError:
            return 0

        return result.count

    def health_status(self) -> str:
        """Return ``ok``, ``collection_missing`` or ``unavailable``; never raises."""
        try:
            exists = self.collection_exists()
        except Exception:  # a health check must not raise
            logger.warning("Qdrant health check failed", exc_info=True)
            return "unavailable"

        return "ok" if exists else "collection_missing"

    def _create_new_collection(self, dimension: int) -> dict[str, Any]:
        logger.info(
            "Creating collection %s (dense=%d, sparse=%s)",
            self.collection_name,
            dimension,
            SPARSE_VECTOR_NAME,
        )

        try:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
                metadata=self._expected_metadata(),
            )
        except UnexpectedResponse:
            # Another process may be creating it at the same moment; the
            # server then answers 409 ("already exists") or even 500 while
            # the collection is still being set up.
            payload_schema = self._wait_for_concurrent_collection(dimension)

            if payload_schema is None:
                raise

            logger.info("Collection %s was created concurrently", self.collection_name)
            return payload_schema

        return {}

    def _wait_for_concurrent_collection(self, dimension: int) -> dict[str, Any] | None:
        """Wait until a collection created by someone else is readable.

        Right after creation the server may already report the collection as
        existing while ``get_collection`` still fails with HTTP 500, so the
        schema read is retried as well. A real schema mismatch is raised.
        """
        for _ in range(CONCURRENT_CREATE_POLLS):
            try:
                if self._client.collection_exists(self.collection_name):
                    return self._validate_schema(dimension).payload_schema or {}
            except (UnexpectedResponse, ResponseHandlingException):
                pass  # still being set up

            time.sleep(CONCURRENT_CREATE_POLL_SECONDS)

        return None

    def _ensure_payload_index(self, payload_schema: dict[str, Any] | None) -> None:
        if _is_local_client(self._client) or "document_id" in (payload_schema or {}):
            return

        self._client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def _validate_schema(self, dimension: int) -> models.CollectionInfo:
        info = self._client.get_collection(self.collection_name)
        params = info.config.params
        problems: list[str] = []

        dense_vectors = params.vectors if isinstance(params.vectors, dict) else {}
        dense = dense_vectors.get(DENSE_VECTOR_NAME)

        if dense is None:
            problems.append(f"named dense vector '{DENSE_VECTOR_NAME}' is missing")
        else:
            if dense.size != dimension:
                problems.append(
                    f"dense size is {dense.size}, embedding model produces {dimension}",
                )
            if dense.distance != models.Distance.COSINE:
                problems.append(f"dense distance is {dense.distance}, expected Cosine")

        sparse = (params.sparse_vectors or {}).get(SPARSE_VECTOR_NAME)

        if sparse is None:
            problems.append(f"sparse vector '{SPARSE_VECTOR_NAME}' is missing")
        elif sparse.modifier != models.Modifier.IDF:
            problems.append("sparse bm25 vector must use the IDF modifier")

        # Vectors from another model (even with the same size) or another BM25
        # text preparation are not comparable with today's query vectors.
        metadata = info.config.metadata or {}

        for key, expected in self._expected_metadata().items():
            if metadata.get(key) != expected:
                problems.append(
                    f"it was built with {key}={metadata.get(key)!r}, "
                    f"current setting is {expected!r}",
                )

        if problems:
            raise CollectionSchemaError(
                f"Collection '{self.collection_name}' does not match the "
                f"CampusLens contract: {'; '.join(problems)}. "
                "Recreate it explicitly (create_collection(recreate=True) or "
                "scripts/index_documents.py --recreate) after checking the data.",
            )

        return info

    def _expected_metadata(self) -> dict[str, Any]:
        return {
            "dense_model": self._embedder.dense_model_name,
            "sparse_model": self._embedder.sparse_model_name,
            "bm25_preprocessing": BM25_PREPROCESSING_VERSION,
        }

    def _require_collection(self) -> None:
        if self._collection_verified:
            return

        if not self.collection_exists():
            raise CollectionNotFoundError(self.collection_name)

        dimension = self._embedder.dense_dimension

        with self._qdrant_errors("validate collection"):
            self._validate_schema(dimension)

        self._collection_verified = True

    # Indexing ---------------------------------------------------------------

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        """Insert or update chunks.

        Point IDs are derived from chunk_id, so writing the same chunk again
        updates the existing point instead of duplicating it. All embeddings
        are computed before anything is written.
        """
        unique_chunks = self._prepare_chunks(chunks)

        if not unique_chunks:
            return

        if not self._collection_verified:
            self.create_collection()

        self._check_chunk_owners(unique_chunks)
        points = self._build_points(unique_chunks)

        try:
            self._upsert_points(points)
        except CollectionNotFoundError:
            logger.warning(
                "Collection %s disappeared during indexing; recreating it",
                self.collection_name,
            )
            self.create_collection()
            self._upsert_points(points)

    def delete_document(self, document_id: str) -> None:
        """Delete every point of one document (no-op if there are none)."""
        document_id = _require_id(document_id, "document_id")

        with self._document_locks_for([document_id]):
            self._delete_points(_document_filter(document_id), "delete document")

    def replace_document_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        """Make the index hold exactly ``chunks`` for each document in them.

        New points are written first and only then are the document's other
        (stale) points removed, so a failure part-way never leaves the
        document missing from search.
        """
        unique_chunks = self._prepare_chunks(chunks)

        if not unique_chunks:
            return

        keep: dict[str, list[str]] = defaultdict(list)

        for chunk in unique_chunks:
            keep[chunk.document_id].append(chunk_point_id(chunk.chunk_id))

        with self._document_locks_for(keep):
            self.upsert_chunks(unique_chunks)

            for document_id, point_ids in keep.items():
                self._delete_points(
                    _document_filter(document_id, exclude_point_ids=point_ids),
                    "remove stale chunks",
                )

    def _prepare_chunks(self, chunks: Sequence[ChunkRecord]) -> list[ChunkRecord]:
        by_chunk_id: dict[str, ChunkRecord] = {}

        for chunk in chunks:
            _require_id(chunk.chunk_id, "chunk_id")
            _require_id(chunk.document_id, "document_id")
            previous = by_chunk_id.get(chunk.chunk_id)

            if previous is not None and previous.document_id != chunk.document_id:
                raise ChunkIdConflictError(
                    f"chunk_id '{chunk.chunk_id}' is used by documents "
                    f"'{previous.document_id}' and '{chunk.document_id}'.",
                )

            by_chunk_id[chunk.chunk_id] = chunk

        if len(by_chunk_id) != len(chunks):
            logger.warning(
                "Ignoring %d duplicate chunk_id values; last occurrence wins",
                len(chunks) - len(by_chunk_id),
            )

        return list(by_chunk_id.values())

    def _check_chunk_owners(self, chunks: list[ChunkRecord]) -> None:
        """Refuse to move an existing chunk_id to a different document."""
        for start in range(0, len(chunks), self.batch_size):
            batch = {
                chunk_point_id(chunk.chunk_id): chunk
                for chunk in chunks[start : start + self.batch_size]
            }

            with self._qdrant_errors("check chunk ids"):
                records = self._client.retrieve(
                    collection_name=self.collection_name,
                    ids=list(batch),
                    with_payload=["document_id"],
                    with_vectors=False,
                )

            for record in records:
                chunk = batch.get(str(record.id))
                owner = (record.payload or {}).get("document_id")

                if chunk is not None and owner not in (None, chunk.document_id):
                    raise ChunkIdConflictError(
                        f"chunk_id '{chunk.chunk_id}' already belongs to document "
                        f"'{owner}', not '{chunk.document_id}'.",
                    )

    def _build_points(self, chunks: list[ChunkRecord]) -> list[models.PointStruct]:
        points = []

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [chunk.text for chunk in batch]
            dense_vectors = self._embedder.encode_documents(texts)
            sparse_vectors = self._embedder.encode_sparse(texts)

            for chunk, dense, sparse in zip(
                batch,
                dense_vectors,
                sparse_vectors,
                strict=True,
            ):
                vector: dict[str, Any] = {DENSE_VECTOR_NAME: dense}

                # A text made only of stop words has no BM25 terms; such a
                # point is still found by dense search.
                if sparse.indices:
                    vector[SPARSE_VECTOR_NAME] = sparse

                points.append(
                    models.PointStruct(
                        id=chunk_point_id(chunk.chunk_id),
                        vector=vector,
                        payload=build_payload(chunk),
                    ),
                )

        return points

    def _upsert_points(self, points: list[models.PointStruct]) -> None:
        for start in range(0, len(points), self.batch_size):
            with self._qdrant_errors("upsert points"):
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points[start : start + self.batch_size],
                    wait=True,
                )

    def _delete_points(self, points_filter: models.Filter, operation: str) -> None:
        try:
            with self._qdrant_errors(operation):
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=points_filter),
                    wait=True,
                )
        except CollectionNotFoundError:
            return

    @contextmanager
    def _document_locks_for(self, document_ids: Iterable[str]) -> Iterator[None]:
        with ExitStack() as stack:
            # Sorted acquisition order prevents deadlocks between replaces.
            for document_id in sorted(set(document_ids)):
                with self._document_locks_guard:
                    lock = self._document_locks.setdefault(
                        document_id,
                        threading.Lock(),
                    )
                stack.enter_context(lock)

            yield

    # Search -----------------------------------------------------------------

    def search_dense(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RetrievedChunk]:
        """Semantic search with the E5 ``query:`` vector."""
        limit = _validate_limit(limit)

        if not has_searchable_text(question):
            return []

        self._require_collection()
        query_vector = self._embedder.encode_query(question)

        points = self._query(
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
        )

        return self._to_retrieved_chunks(points, RetrievalMethod.DENSE)

    def search_bm25(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RetrievedChunk]:
        """Keyword search with BM25 sparse vectors (IDF computed by Qdrant)."""
        limit = _validate_limit(limit)

        if not has_searchable_text(question):
            return []

        self._require_collection()
        query_vector = self._embedder.encode_sparse_query(question)

        if not query_vector.indices:
            return []

        points = self._query(
            query=query_vector,
            using=SPARSE_VECTOR_NAME,
            limit=limit,
        )

        return self._to_retrieved_chunks(points, RetrievalMethod.BM25)

    def search_hybrid_rrf(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RetrievedChunk]:
        """Dense and BM25 candidates fused with Reciprocal Rank Fusion."""
        limit = _validate_limit(limit)

        if not has_searchable_text(question):
            return []

        self._require_collection()
        dense_vector = self._embedder.encode_query(question)
        sparse_vector = self._embedder.encode_sparse_query(question)

        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=max(limit, self._settings.dense_top_k),
            ),
        ]

        if sparse_vector.indices:
            prefetch.append(
                models.Prefetch(
                    query=sparse_vector,
                    using=SPARSE_VECTOR_NAME,
                    limit=max(limit, self._settings.bm25_top_k),
                ),
            )

        points = self._query(
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        return self._to_retrieved_chunks(points, RetrievalMethod.HYBRID)

    def _query(self, **kwargs: Any) -> list[models.ScoredPoint]:
        with self._qdrant_errors("search"):
            response = self._client.query_points(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                **kwargs,
            )

        return response.points

    def _to_retrieved_chunks(
        self,
        points: list[models.ScoredPoint],
        method: RetrievalMethod,
    ) -> list[RetrievedChunk]:
        chunks = []

        for point in points:
            payload = point.payload or {}

            try:
                chunk = RetrievedChunk(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    document_title=payload["document_title"],
                    source_url=payload["source_url"],
                    page_number=payload["page_number"],
                    section_title=payload.get("section_title"),
                    text=payload["text"],
                    retrieval_method=method,
                    retrieval_score=float(point.score),
                    rerank_score=None,
                )
            except (KeyError, ValidationError) as exc:
                raise VectorStoreError(
                    f"Point {point.id} in '{self.collection_name}' has an "
                    f"invalid payload: {exc}",
                ) from exc

            chunks.append(chunk)

        return chunks

    # Error mapping ----------------------------------------------------------

    @contextmanager
    def _qdrant_errors(
        self,
        operation: str,
        *,
        not_found_is_missing: bool = True,
    ) -> Iterator[None]:
        """Translate every client error into one of our ApplicationErrors."""
        with self._client_lock:
            try:
                yield
            except ApplicationError:
                raise
            except ResourceExhaustedResponse as exc:
                raise self._unavailable(operation, f"rate limited: {exc}") from exc
            except UnexpectedResponse as exc:
                status = exc.status_code or 0

                if status == 404 and not_found_is_missing:
                    self._collection_verified = False
                    raise CollectionNotFoundError(self.collection_name) from exc

                if status in (404, 429) or status >= 500:
                    raise self._unavailable(operation, f"HTTP {status}: {exc}") from exc

                raise self._rejected(operation, exc) from exc
            except ResponseHandlingException as exc:
                if isinstance(exc.source, ValidationError):
                    raise self._rejected(
                        operation,
                        "unreadable response (client/server version mismatch?): "
                        f"{exc.source}",
                    ) from exc

                raise self._unavailable(operation, repr(exc.source)) from exc
            except (
                httpx.TransportError,
                ConnectionError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                raise self._unavailable(operation, repr(exc)) from exc
            except ValueError as exc:
                # Local mode reports a missing collection as ValueError.
                if not_found_is_missing and "not found" in str(exc).lower():
                    self._collection_verified = False
                    raise CollectionNotFoundError(self.collection_name) from exc

                raise self._rejected(operation, exc) from exc
            except (QdrantException, AssertionError, RuntimeError) as exc:
                raise self._rejected(operation, exc) from exc

    def _unavailable(self, operation: str, detail: str) -> VectorStoreUnavailableError:
        message = (
            f"Qdrant at {self._settings.qdrant_url} unavailable during "
            f"{operation}: {detail}"
        )
        logger.error(message)
        return VectorStoreUnavailableError(message)

    def _rejected(self, operation: str, detail: object) -> VectorStoreError:
        message = f"Qdrant rejected {operation}: {detail}"
        logger.error(message)
        return VectorStoreError(message)


class QdrantVectorStore:
    """Async implementation of ``backend.app.ports.vector_store.VectorStore``.

    Embedding (CPU) and Qdrant calls (network I/O) are blocking, so they run
    in a worker thread and never block the event loop.
    """

    def __init__(self, store: QdrantStore | None = None) -> None:
        self._store = store or get_qdrant_store()

    @property
    def store(self) -> QdrantStore:
        return self._store

    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        await asyncio.to_thread(self._store.upsert_chunks, chunks)

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(self._store.delete_document, document_id)

    async def search_bm25(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._store.search_bm25, question, limit)

    async def search_dense(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._store.search_dense, question, limit)

    async def search_hybrid_rrf(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._store.search_hybrid_rrf, question, limit)

    # Extras beyond the port -------------------------------------------------

    async def replace_document_chunks(self, chunks: list[ChunkRecord]) -> None:
        await asyncio.to_thread(self._store.replace_document_chunks, chunks)

    async def create_collection(self, recreate: bool = False) -> None:
        await asyncio.to_thread(self._store.create_collection, recreate)

    async def warm_up(self) -> None:
        await asyncio.to_thread(self._store.warm_up)

    async def health_status(self) -> str:
        return await asyncio.to_thread(self._store.health_status)


_default_store: QdrantStore | None = None
_default_vector_store: QdrantVectorStore | None = None
_default_lock = threading.Lock()


def get_qdrant_store() -> QdrantStore:
    """Process-wide synchronous engine (scripts and offline evaluation)."""
    global _default_store

    if _default_store is None:
        with _default_lock:
            if _default_store is None:
                _default_store = QdrantStore()

    return _default_store


def get_vector_store() -> QdrantVectorStore:
    """FastAPI dependency returning the async ``VectorStore`` implementation."""
    global _default_vector_store

    if _default_vector_store is None:
        store = get_qdrant_store()

        with _default_lock:
            if _default_vector_store is None:
                _default_vector_store = QdrantVectorStore(store)

    return _default_vector_store
