from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_run_mock_evaluation() -> None:
    response = client.post(
        "/api/evaluations/run",
        json={
            "method": "hybrid_rerank",
            "split": "validation",
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert body["method"] == "hybrid_rerank"
    assert body["split"] == "validation"
    assert body["status"] == "completed"
    assert body["is_mock"] is True
    assert "run_id" in body


def test_list_evaluation_results() -> None:
    client.post(
        "/api/evaluations/run",
        json={
            "method": "dense",
            "split": "validation",
        },
    )

    response = client.get("/api/evaluations/results")

    assert response.status_code == 200

    body = response.json()

    assert body["total"] >= 1
    assert len(body["results"]) >= 1
    assert body["results"][0]["metrics"] is not None
    assert body["results"][0]["is_mock"] is True


def test_rejects_unknown_evaluation_method() -> None:
    response = client.post(
        "/api/evaluations/run",
        json={
            "method": "unknown_method",
            "split": "validation",
        },
    )

    assert response.status_code == 422


def test_rejects_unknown_evaluation_split() -> None:
    response = client.post(
        "/api/evaluations/run",
        json={
            "method": "dense",
            "split": "development",
        },
    )

    assert response.status_code == 422
