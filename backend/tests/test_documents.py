from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF"


def upload_test_document():
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


def test_upload_document() -> None:
    response = upload_test_document()

    assert response.status_code == 201

    body = response.json()

    assert body["title"] == "Önlisans ve Lisans Yönetmeliği"
    assert body["file_name"] == "regulation.pdf"
    assert body["status"] == "uploaded"
    assert body["chunk_count"] == 0


def test_list_documents() -> None:
    upload_test_document()

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()["total"] >= 1


def test_rejects_non_pdf_file() -> None:
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


def test_delete_document() -> None:
    upload_response = upload_test_document()
    document_id = upload_response.json()["document_id"]

    delete_response = client.delete(
        f"/api/documents/{document_id}",
    )

    assert delete_response.status_code == 204


def test_delete_unknown_document() -> None:
    response = client.delete(
        "/api/documents/00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404
