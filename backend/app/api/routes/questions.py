from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.ports.question_pipeline import QuestionPipeline
from backend.app.schemas.questions import AskRequest, AskResponse
from backend.app.services.question_service import (
    ask_question,
    get_question_pipeline,
)

router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
)


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    pipeline: Annotated[
        QuestionPipeline,
        Depends(get_question_pipeline),
    ],
) -> AskResponse:
    return await ask_question(request, pipeline)
