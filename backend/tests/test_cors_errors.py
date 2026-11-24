from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

FRONTEND_ORIGIN = "http://127.0.0.1:5500"
UNTRUSTED_ORIGIN = "http://example.com"


def build_test_app() -> FastAPI:
    settings = Settings(
        app_env="development",
        frontend_origin=FRONTEND_ORIGIN,
    )
    app = create_app(settings)

    @app.get("/api/test/unexpected-error")
    async def raise_unexpected_error() -> dict[str, str]:
        raise RuntimeError("Test server failure.")

    return app


def test_unexpected_error_includes_cors_for_allowed_origin() -> None:
    client = TestClient(
        build_test_app(),
        raise_server_exceptions=False,
    )

    response = client.get(
        "/api/test/unexpected-error",
        headers={"Origin": FRONTEND_ORIGIN},
    )

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == FRONTEND_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "Origin" in response.headers["vary"]

    body = response.json()

    assert body["error"] == "internal_server_error"
    assert body["message"] == "Beklenmeyen bir sunucu hatası oluştu."
    assert body["request_id"] != "unknown"
    assert response.headers["x-request-id"] == body["request_id"]


def test_unexpected_error_rejects_untrusted_origin() -> None:
    client = TestClient(
        build_test_app(),
        raise_server_exceptions=False,
    )

    response = client.get(
        "/api/test/unexpected-error",
        headers={"Origin": UNTRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert "access-control-allow-origin" not in response.headers
