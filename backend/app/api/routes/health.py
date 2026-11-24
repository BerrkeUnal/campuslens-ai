import logging
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.db.database import SQLiteDatabase, get_database
from backend.app.rag.qdrant_store import (
    QdrantVectorStore,
    get_vector_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

HealthCheck = Callable[[], Awaitable[str]]


async def safe_health_status(
    service_name: str,
    check: HealthCheck,
) -> str:
    try:
        return await check()
    except Exception:
        logger.warning(
            "%s health check failed",
            service_name,
            exc_info=True,
        )
        return "unavailable"


@router.get("/health")
async def health_check(
    database: Annotated[
        SQLiteDatabase,
        Depends(get_database),
    ],
    vector_store: Annotated[
        QdrantVectorStore,
        Depends(get_vector_store),
    ],
) -> dict[str, str]:
    database_status = await safe_health_status(
        "SQLite",
        database.health_status,
    )
    qdrant_status = await safe_health_status(
        "Qdrant",
        vector_store.health_status,
    )

    overall_status = (
        "ok" if database_status == "ok" and qdrant_status == "ok" else "degraded"
    )

    return {
        "status": overall_status,
        "backend": "ok",
        "database": database_status,
        "qdrant": qdrant_status,
    }
