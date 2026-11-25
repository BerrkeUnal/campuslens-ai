from collections.abc import Awaitable, Callable
from inspect import iscoroutinefunction
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from backend.app.ports.question_pipeline import QuestionPipeline
from backend.app.rag.model_readiness_guard import ensure_models_ready
from backend.app.schemas.questions import AskRequest, AskResponse
from backend.app.services.question_service import (
    ask_question,
    get_question_pipeline,
)

router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
)

PipelineResolver = Callable[[], Awaitable[QuestionPipeline]]


async def get_question_pipeline_resolver(
    request: Request,
) -> PipelineResolver:
    """Defer pipeline construction until validation and readiness checks."""

    async def resolve() -> QuestionPipeline:
        # Preserve existing parameterless pipeline factory overrides.
        factory = request.app.dependency_overrides.get(
            get_question_pipeline,
            get_question_pipeline,
        )

        if iscoroutinefunction(factory):
            return await factory()

        return await run_in_threadpool(factory)

    return resolve


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    http_request: Request,
    resolver: Annotated[
        PipelineResolver,
        Depends(get_question_pipeline_resolver),
    ],
) -> AskResponse:
    ensure_models_ready(
        http_request.app.state.model_readiness,
    )

    pipeline = await resolver()

    return await ask_question(request, pipeline)
