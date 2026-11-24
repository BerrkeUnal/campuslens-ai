from asyncio import to_thread
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import Request

from backend.app.core.exceptions import ServiceUnavailableError

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
"""


class SQLiteDatabase:
    def __init__(self, url: str) -> None:
        scheme, separator, file_path = url.partition(":///")
        if not separator or scheme not in {"sqlite", "sqlite+aiosqlite"} or not file_path:
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


def get_database(request: Request) -> SQLiteDatabase:
    return request.app.state.database
