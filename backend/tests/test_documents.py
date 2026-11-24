from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'documents.db').as_posix()}")
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def upload_test_document(client: TestClient):
    return client.post(
        "/api/documents/upload",
        files={
            "file": (
                "regulation.pdf",
                PDF_CONTENT,
                "application/pdf",
            ),
        },
        data={
            "title": "Önlisans ve Lisans Yönetmeliği",
            "source_url": "https://www.example.edu.tr/regulation",
        },
    )


def test_upload_document(client: TestClient) -> None:
    response = upload_test_document(client)

    assert response.status_code == 201

    body = response.json()

    assert body["title"] == "Önlisans ve Lisans Yönetmeliği"
    assert body["file_name"] == "regulation.pdf"
    assert body["status"] == "uploaded"
    assert body["chunk_count"] == 0


def test_list_documents(client: TestClient) -> None:
    upload_test_document(client)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()["total"] >= 1


def test_rejects_non_pdf_file(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"not a pdf",
                "text/plain",
            ),
        },
        data={
            "title": "Test Belgesi",
            "source_url": "https://www.example.edu.tr/test",
        },
    )

    assert response.status_code == 415


def test_delete_document(client: TestClient) -> None:
    upload_response = upload_test_document(client)
    document_id = upload_response.json()["document_id"]

    delete_response = client.delete(
        f"/api/documents/{document_id}",
    )

    assert delete_response.status_code == 204


def test_delete_unknown_document(client: TestClient) -> None:
    response = client.delete(
        "/api/documents/00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404
