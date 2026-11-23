import json
import logging

import pytest
from fastapi import Request

from backend.app.core.error_handlers import (
    application_error_handler,
)
from backend.app.core.exceptions import ServiceUnavailableError


def build_request() -> Request:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
        }
    )
    request.state.request_id = "test-request-id"

    return request


@pytest.mark.asyncio
async def test_application_error_handler_logs_original_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = build_request()

    try:
        try:
            raise RuntimeError("Upstream service failed.")
        except RuntimeError as cause:
            raise ServiceUnavailableError("LLM") from cause
    except ServiceUnavailableError as exception:
        with caplog.at_level(
            logging.WARNING,
            logger="backend.app.core.error_handlers",
        ):
            response = await application_error_handler(
                request,
                exception,
            )

    assert response.status_code == 503

    body = json.loads(response.body)

    assert body["error"] == "service_unavailable"
    assert body["request_id"] == "test-request-id"
    assert "LLM" in body["message"]
    assert "Upstream service failed." not in body["message"]

    record = caplog.records[-1]

    assert record.exc_info is not None
    logged_exception = record.exc_info[1]

    assert isinstance(
        logged_exception,
        ServiceUnavailableError,
    )
    assert isinstance(
        logged_exception.__cause__,
        RuntimeError,
    )
    assert "request_id=test-request-id" in record.getMessage()
    assert "code=service_unavailable" in record.getMessage()
