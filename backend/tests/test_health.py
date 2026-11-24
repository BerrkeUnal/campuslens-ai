import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.db.database import get_database
from backend.app.main import create_app
from backend.app.rag.qdrant_store import get_vector_store


class HealthStub:
    def __init__(self, result: str | Exception) -> None:
        self.result = result

    async def health_status(self) -> str:
        if isinstance(self.result, Exception):
            raise self.result

        return self.result


def build_client(
    database_result: str | Exception,
    qdrant_result: str | Exception,
) -> TestClient:
    app = create_app(Settings(app_env="testing"))

    app.dependency_overrides[get_database] = lambda: HealthStub(database_result)
    app.dependency_overrides[get_vector_store] = lambda: HealthStub(qdrant_result)

    return TestClient(
        app,
        raise_server_exceptions=False,
    )


def test_health_check_reports_ready_services() -> None:
    client = build_client("ok", "ok")

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "backend": "ok",
        "database": "ok",
        "qdrant": "ok",
    }


@pytest.mark.parametrize(
    ("database_status", "qdrant_status"),
    [
        ("unavailable", "ok"),
        ("ok", "unavailable"),
        ("ok", "collection_missing"),
    ],
)
def test_health_check_reports_degraded_services(
    database_status: str,
    qdrant_status: str,
) -> None:
    client = build_client(database_status, qdrant_status)

    response = client.get("/api/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "degraded"
    assert body["database"] == database_status
    assert body["qdrant"] == qdrant_status


def test_health_check_converts_unexpected_failure_to_unavailable() -> None:
    client = build_client(
        RuntimeError("Database health check failed."),
        "ok",
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "backend": "ok",
        "database": "unavailable",
        "qdrant": "ok",
    }
