from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.documents import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
)
from backend.app.services.document_service import (
    DocumentService,
    get_document_service,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form(min_length=2, max_length=200)],
    source_url: Annotated[str, Form(min_length=8, max_length=2048)],
    document_service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentResponse:
    file_name = file.filename or ""

    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Yalnızca PDF dosyaları yüklenebilir.",
        )

    maximum_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read(maximum_bytes + 1)

    if len(content) > maximum_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(f"Dosya boyutu en fazla {settings.max_upload_mb} MB olabilir."),
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Yüklenen dosya geçerli bir PDF değil.",
        )

    metadata = DocumentCreate(
        title=title,
        source_url=source_url,
    )

    return await document_service.create_document(
        metadata=metadata,
        file_name=file_name,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    document_service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> DocumentListResponse:
    return await document_service.list_documents()


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: UUID,
    document_service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> Response:
    deleted = await document_service.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Belge bulunamadı.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
