import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.core.exceptions import ApplicationError

logger = logging.getLogger(__name__)


def get_request_id(request: Request) -> str:
    return getattr(
        request.state,
        "request_id",
        "unknown",
    )


async def application_error_handler(
    request: Request,
    exception: ApplicationError,
) -> JSONResponse:
    request_id = get_request_id(request)

    logger.warning(
        "Application error request_id=%s code=%s message=%s",
        request_id,
        exception.code,
        exception.message,
        exc_info=(
            type(exception),
            exception,
            exception.__traceback__,
        ),
    )

    return JSONResponse(
        status_code=exception.status_code,
        content={
            "error": exception.code,
            "message": exception.message,
            "request_id": request_id,
            "details": None,
        },
    )


async def validation_error_handler(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    request_id = get_request_id(request)

    details: list[dict[str, Any]] = []

    for error in exception.errors():
        field_parts = error.get("loc", ())
        field = ".".join(str(part) for part in field_parts)

        details.append(
            {
                "field": field,
                "type": error.get("type", "validation_error"),
            },
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": "validation_error",
            "message": "İstek alanlarından biri geçersiz.",
            "request_id": request_id,
            "details": details,
        },
    )


async def unexpected_error_handler(
    request: Request,
    exception: Exception,
) -> JSONResponse:
    request_id = get_request_id(request)

    logger.error(
        "Unhandled error request_id=%s",
        request_id,
        exc_info=(
            type(exception),
            exception,
            exception.__traceback__,
        ),
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "Beklenmeyen bir sunucu hatası oluştu.",
            "request_id": request_id,
            "details": None,
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        ApplicationError,
        application_error_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )
    app.add_exception_handler(
        Exception,
        unexpected_error_handler,
    )
