from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_ask_question_returns_mock_response() -> None:
    response = client.post(
        "/api/questions/ask",
        json={
            "question": "Bütünleme sınavına kimler girebilir?",
            "method": "hybrid_rerank",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["method"] == "hybrid_rerank"
    assert body["insufficient_context"] is False
    assert "[S1]" in body["answer"]
    assert body["sources"][0]["source_id"] == "S1"
    assert "metrics" in body


def test_ask_question_rejects_short_question() -> None:
    response = client.post(
        "/api/questions/ask",
        json={
            "question": "a",
            "method": "hybrid_rerank",
        },
    )

    assert response.status_code == 422


def test_ask_question_rejects_unknown_method() -> None:
    response = client.post(
        "/api/questions/ask",
        json={
            "question": "Bütünleme sınavı ne zaman?",
            "method": "unknown_method",
        },
    )

    assert response.status_code == 422
