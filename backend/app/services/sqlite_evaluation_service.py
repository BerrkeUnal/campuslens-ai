import logging
from asyncio import Lock
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends

from backend.app.core.exceptions import ApplicationError
from backend.app.db.database import SQLiteDatabase, get_database
from backend.app.db.evaluation_repository import SQLiteEvaluationRepository
from backend.app.ports.vector_store import VectorStore
from backend.app.rag.hybrid_rerank import HybridRerankSearch
from backend.app.rag.pipeline import RAGPipeline
from backend.app.rag.qdrant_store import get_vector_store
from backend.app.rag.reranker import get_reranker
from backend.app.rag.retriever import Retriever
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
    ResolvedEvaluationCase,
    load_evaluation_cases,
)
from backend.app.services.retrieval_evaluation import RetrievalEvaluationRunner

logger = logging.getLogger(__name__)


class SQLiteEvaluationService:
    def __init__(
        self,
        repository: SQLiteEvaluationRepository,
        runner_factory: Callable[[], RetrievalEvaluationRunner],
        cases_path: Path = CASES_PATH,
        catalog_path: Path = CATALOG_PATH,
    ) -> None:
        self._repository = repository
        self._runner_factory = runner_factory
        self._cases_path = cases_path
        self._catalog_path = catalog_path

    async def prepare_run(
        self, request: EvaluationRunRequest
    ) -> tuple[EvaluationRunResponse, list[ResolvedEvaluationCase]]:
        try:
            cases = [
                item
                for item in load_evaluation_cases(self._cases_path, self._catalog_path)
                if item.case.split == request.split
            ]
            if not cases:
                raise EvaluationDataError(f"{request.split.value} bölümünde test sorusu yok.")
        except EvaluationDataError as exc:
            run = await self._repository.start_run(request, total_cases=0)
            await self._repository.fail_run(run.run_id, str(exc))
            return run.model_copy(update={"status": EvaluationStatus.FAILED}), []

        run = await self._repository.start_run(request, total_cases=len(cases))
        return run, cases

    async def execute_run(
        self,
        run: EvaluationRunResponse,
        cases: list[ResolvedEvaluationCase],
        lock: Lock | None = None,
    ) -> EvaluationRunResponse:
        try:
            async with lock if lock is not None else nullcontext():
                runner = self._runner_factory()
                if runner.is_mock:
                    raise ValueError("Sahte arama sonuçları kalıcı değerlendirmeye yazılamaz.")
                evaluation = await runner.evaluate(run.method, cases)
                await self._repository.complete_run(run.run_id, evaluation)
        except Exception as exc:  # Persist the failure of the external evaluation job.
            logger.exception("Evaluation run %s failed", run.run_id)
            message = (
                str(exc)
                if isinstance(exc, (ApplicationError, EvaluationDataError))
                else "Değerlendirme tamamlanamadı. Ayrıntılar sunucu kayıtlarında."
            )
            await self._repository.fail_run(run.run_id, message)
            return run.model_copy(update={"status": EvaluationStatus.FAILED})

        return run.model_copy(update={"status": EvaluationStatus.COMPLETED})

    async def run_evaluation(self, request: EvaluationRunRequest) -> EvaluationRunResponse:
        run, cases = await self.prepare_run(request)
        if run.status == EvaluationStatus.RUNNING:
            return await self.execute_run(run, cases)
        return run

    async def list_results(self) -> EvaluationResultListResponse:
        results = await self._repository.list_results()
        return EvaluationResultListResponse(results=results, total=len(results))

    async def get_result(self, run_id: UUID) -> EvaluationResultResponse | None:
        return await self._repository.get_result(run_id)


def create_evaluation_runner(store: VectorStore) -> RetrievalEvaluationRunner:
    retriever = Retriever(store)
    reranked_search = HybridRerankSearch(retriever, get_reranker())
    pipeline = RAGPipeline(retriever, reranked_search)
    return RetrievalEvaluationRunner(
        retriever,
        reranked_search.search,
        is_mock=False,
        answer_with_context=pipeline.answer_with_context,
    )


def get_evaluation_runner() -> Callable[[], RetrievalEvaluationRunner]:
    return lambda: create_evaluation_runner(get_vector_store())


def get_sqlite_evaluation_service(
    database: Annotated[SQLiteDatabase, Depends(get_database)],
    runner_factory: Annotated[Callable[[], RetrievalEvaluationRunner], Depends(get_evaluation_runner)],
) -> SQLiteEvaluationService:
    return SQLiteEvaluationService(SQLiteEvaluationRepository(database), runner_factory)
