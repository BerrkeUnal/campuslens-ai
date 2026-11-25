from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status

from backend.app.db.database import SQLiteDatabase, get_database
from backend.app.db.evaluation_repository import SQLiteEvaluationRepository
from backend.app.schemas.evaluations import (
    EvaluationResultListResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationStatus,
)
from backend.app.services.sqlite_evaluation_service import (
    SQLiteEvaluationService,
    get_sqlite_evaluation_service,
)

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_evaluation(
    payload: EvaluationRunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    evaluation_service: Annotated[SQLiteEvaluationService, Depends(get_sqlite_evaluation_service)],
) -> EvaluationRunResponse:
    run, cases = await evaluation_service.prepare_run(payload)
    if run.status == EvaluationStatus.RUNNING:
        background_tasks.add_task(
            evaluation_service.execute_run, run, cases, request.app.state.evaluation_lock
        )
    return run


@router.get("/results", response_model=EvaluationResultListResponse)
async def list_evaluation_results(
    database: Annotated[SQLiteDatabase, Depends(get_database)],
) -> EvaluationResultListResponse:
    results = await SQLiteEvaluationRepository(database).list_results()
    return EvaluationResultListResponse(results=results, total=len(results))
