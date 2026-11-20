from typing import Annotated

from fastapi import APIRouter, Depends, status

from backend.app.schemas.evaluations import (
    EvaluationResultListResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
)
from backend.app.services.evaluation_service import (
    EvaluationService,
    get_evaluation_service,
)

router = APIRouter(
    prefix="/evaluations",
    tags=["Evaluations"],
)


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_evaluation(
    request: EvaluationRunRequest,
    evaluation_service: Annotated[
        EvaluationService,
        Depends(get_evaluation_service),
    ],
) -> EvaluationRunResponse:
    return await evaluation_service.run_evaluation(request)


@router.get(
    "/results",
    response_model=EvaluationResultListResponse,
)
async def list_evaluation_results(
    evaluation_service: Annotated[
        EvaluationService,
        Depends(get_evaluation_service),
    ],
) -> EvaluationResultListResponse:
    return await evaluation_service.list_results()
