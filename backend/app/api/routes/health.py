from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "backend": "ok",
        "database": "not_checked",
        "qdrant": "not_checked",
    }
