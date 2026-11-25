import logging
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends

from backend.app.core.exceptions import ApplicationError
from backend.app.db.database import SQLiteDatabase, get_database
from backend.app.db.evaluation_repository import SQLiteEvaluationRepository
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.hybrid_rerank import search_hybrid_rerank
from backend.app.rag.qdrant_store import get_vector_store
from backend.app.schemas.evaluations import (
    EvaluationResultListResponse,
    EvaluationResultResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationStatus,
)
from backend.app.services.evaluation_cases import (
    CASES_PATH,
    CATALOG_PATH,
    EvaluationDataError,
    load_evaluation_cases,
)
from backend.app.services.retrieval_evaluation import RetrievalEvaluationRunner, Search

logger = logging.getLogger(__name__)


class SQLiteEvaluationService:
    def __init__(
        self,
        repository: SQLiteEvaluationRepository,
        runner: RetrievalEvaluationRunner,
        cases_path: Path = CASES_PATH,
        catalog_path: Path = CATALOG_PATH,
    ) -> None:
        self._repository = repository
        self._runner = runner
        self._cases_path = cases_path
        self._catalog_path = catalog_path

    async def run_evaluation(
        self, request: EvaluationRunRequest
    ) -> EvaluationRunResponse:
        if self._runner.is_mock:
            raise ValueError("Sahte arama sonuçları kalıcı değerlendirmeye yazılamaz.")

        try:
            cases = [
                item
                for item in load_evaluation_cases(self._cases_path, self._catalog_path)
                if item.case.split == request.split
            ]
        except EvaluationDataError as exc:
            run = await self._repository.start_run(request, total_cases=0)
            await self._repository.fail_run(run.run_id, str(exc))
            return run.model_copy(update={"status": EvaluationStatus.FAILED})

        run = await self._repository.start_run(request, total_cases=len(cases))
        try:
            if not cases:
                raise EvaluationDataError(
                    f"{request.split.value} bölümünde test sorusu yok."
                )
            evaluation = await self._runner.evaluate(request.method, cases)
            await self._repository.complete_run(run.run_id, evaluation)
        except (
            Exception
        ) as exc:  # Save failed runs, including errors from injected retrieval.
            logger.exception("Evaluation run %s failed", run.run_id)
            message = (
                str(exc)
                if isinstance(exc, (ApplicationError, EvaluationDataError))
                else "Değerlendirme tamamlanamadı. Ayrıntılar sunucu kayıtlarında."
            )
            await self._repository.fail_run(run.run_id, message)
            return run.model_copy(update={"status": EvaluationStatus.FAILED})

        return run.model_copy(update={"status": EvaluationStatus.COMPLETED})

    async def list_results(self) -> EvaluationResultListResponse:
        results = await self._repository.list_results()
        return EvaluationResultListResponse(results=results, total=len(results))

    async def get_result(self, run_id: UUID) -> EvaluationResultResponse | None:
        return await self._repository.get_result(run_id)


def get_hybrid_rerank_search() -> Search:
    """Return the real hybrid reranking adapter."""
    return search_hybrid_rerank


def get_sqlite_evaluation_service(
    database: Annotated[SQLiteDatabase, Depends(get_database)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    hybrid_rerank: Annotated[Search | None, Depends(get_hybrid_rerank_search)],
) -> SQLiteEvaluationService:
    return SQLiteEvaluationService(
        SQLiteEvaluationRepository(database),
        RetrievalEvaluationRunner(store, hybrid_rerank, is_mock=False),
    )
