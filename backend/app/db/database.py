import logging
from asyncio import to_thread
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import Request

from backend.app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    file_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('uploaded', 'processing', 'indexed', 'failed')),
    page_count INTEGER CHECK (page_count >= 1),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    created_at TEXT NOT NULL,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    method TEXT NOT NULL,
    split TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    total_cases INTEGER NOT NULL CHECK (total_cases >= 0),
    recall_at_5 REAL CHECK (recall_at_5 BETWEEN 0 AND 1),
    mrr_at_5 REAL CHECK (mrr_at_5 BETWEEN 0 AND 1),
    average_latency_ms REAL CHECK (average_latency_ms >= 0),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error_message TEXT,
    is_mock INTEGER NOT NULL CHECK (is_mock IN (0, 1))
);
CREATE TABLE IF NOT EXISTS evaluation_case_results (
    run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    case_id TEXT NOT NULL,
    source_rank INTEGER CHECK (source_rank BETWEEN 1 AND 5),
    source_hit INTEGER NOT NULL CHECK (source_hit = (source_rank IS NOT NULL)),
    latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
    PRIMARY KEY (run_id, case_id)
);
CREATE TABLE IF NOT EXISTS evaluation_abstentions (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    insufficient_context INTEGER CHECK (insufficient_context IN (0, 1)),
    answer TEXT,
    error_message TEXT,
    CHECK (
        (insufficient_context IS NOT NULL AND answer IS NOT NULL AND error_message IS NULL)
        OR (insufficient_context IS NULL AND answer IS NULL AND error_message IS NOT NULL)
    ),
    PRIMARY KEY (run_id, case_id),
    FOREIGN KEY (run_id, case_id) REFERENCES evaluation_case_results(run_id, case_id)
        ON DELETE CASCADE
);
"""


class SQLiteDatabase:
    def __init__(self, url: str) -> None:
        scheme, separator, file_path = url.partition(":///")
        if (
            not separator
            or scheme not in {"sqlite", "sqlite+aiosqlite"}
            or not file_path
        ):
            raise ValueError("DATABASE_URL must point to a SQLite file.")
        if file_path == ":memory:":
            raise ValueError("Use a temporary SQLite file for an isolated database.")
        self.path = Path(file_path).resolve()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        try:
            async with aiosqlite.connect(self.path, timeout=5) as connection:
                connection.row_factory = aiosqlite.Row
                async with connection.execute("PRAGMA foreign_keys = ON"):
                    pass
                yield connection
        except aiosqlite.Error as exc:
            raise ServiceUnavailableError("SQLite") from exc

    async def initialize(self) -> None:
        try:
            await to_thread(self.path.parent.mkdir, parents=True, exist_ok=True)
        except OSError as exc:
            raise ServiceUnavailableError("SQLite") from exc

        async with self.connect() as connection:
            await connection.executescript(SCHEMA)
            await connection.commit()

    async def health_status(self) -> str:
        """Return the SQLite availability without raising an exception."""
        try:
            async with self.connect() as connection, connection.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()

            if row is not None and row[0] == 1:
                return "ok"

        except Exception:
            logger.warning(
                "SQLite health check failed",
                exc_info=True,
            )

        return "unavailable"


def get_database(request: Request) -> SQLiteDatabase:
    return request.app.state.database
