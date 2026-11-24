from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.core.exceptions import ServiceUnavailableError
from backend.app.db.database import SQLiteDatabase
from backend.app.db.evaluation_repository import SQLiteEvaluationRepository
from backend.app.schemas.evaluations import (
    EvaluationRunRequest,
    EvaluationSplit,
    EvaluationStatus,
)
from backend.app.schemas.retrieval import RetrievalMethod
from backend.app.services.evaluation_metrics import (
    RetrievalCaseResult,
    summarize_retrieval,
)
from backend.app.services.retrieval_evaluation import RetrievalEvaluation
from backend.app.services.sqlite_evaluation_service import SQLiteEvaluationService

REQUEST = EvaluationRunRequest(method=RetrievalMethod.DENSE)


@pytest.fixture
def database(tmp_path: Path) -> SQLiteDatabase:
    return SQLiteDatabase(f"sqlite:///{(tmp_path / 'results.db').as_posix()}")


@pytest.mark.asyncio
async def test_run_and_case_results_persist(database: SQLiteDatabase) -> None:
    await database.initialize()
    repository = SQLiteEvaluationRepository(database)
    cases = [RetrievalCaseResult("eval-001", 5, 10), RetrievalCaseResult("eval-002", None, 30)]
    run = await repository.start_run(REQUEST, total_cases=len(cases))
    await repository.complete_run(run.run_id, RetrievalEvaluation(cases, summarize_retrieval(cases)))

    reopened = SQLiteDatabase(f"sqlite:///{database.path.as_posix()}")
    await reopened.initialize()
    repository = SQLiteEvaluationRepository(reopened)
    result = await repository.get_result(run.run_id)
    assert result is not None
    assert result.status is EvaluationStatus.COMPLETED
    assert result.completed_at is not None
    assert result.is_mock is False
    assert result.metrics is not None
    assert result.metrics.total_cases == 2
    assert result.metrics.recall_at_5 == 0.5
    assert result.metrics.mrr_at_5 == 0.1
    assert result.metrics.average_latency_ms == 20
    assert result.metrics.answer_correctness is None
    assert result.metrics.hallucination_rate is None
    assert result.metrics.abstention_accuracy is None
    assert result.metrics.total_cost_usd is None
    assert await repository.list_results() == [result]
    assert await repository.list_case_results(run.run_id) == cases
    assert await repository.get_result(uuid4()) is None


@pytest.mark.asyncio
async def test_database_error_rolls_back_run_and_cases(database: SQLiteDatabase) -> None:
    await database.initialize()
    repository = SQLiteEvaluationRepository(database)
    cases = [RetrievalCaseResult("duplicate", 1, 10), RetrievalCaseResult("duplicate", 1, 10)]
    run = await repository.start_run(REQUEST, total_cases=len(cases))

    with pytest.raises(ServiceUnavailableError):
        await repository.complete_run(
            run.run_id, RetrievalEvaluation(cases, summarize_retrieval(cases))
        )

    assert await repository.list_case_results(run.run_id) == []
    result = await repository.get_result(run.run_id)
    assert result is not None
    assert result.status is EvaluationStatus.RUNNING
    assert result.metrics is None
    await repository.fail_run(run.run_id, "SQLite kaydı tamamlanamadı.")
    result = await repository.get_result(run.run_id)
    assert result is not None
    assert result.status is EvaluationStatus.FAILED
    assert result.error_message == "SQLite kaydı tamamlanamadı."


class UnusedRunner:
    is_mock = False

    async def evaluate(self, *_):
        raise AssertionError("Retrieval should not start with an empty or invalid data set.")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_data", [False, True])
async def test_service_records_dataset_errors(
    database: SQLiteDatabase, tmp_path: Path, invalid_data: bool
) -> None:
    await database.initialize()
    options = {}
    if invalid_data:
        path = tmp_path / "bad.jsonl"
        path.write_text("{", encoding="utf-8")
        options["cases_path"] = path
    service = SQLiteEvaluationService(SQLiteEvaluationRepository(database), UnusedRunner(), **options)
    run = await service.run_evaluation(
        EvaluationRunRequest(method=RetrievalMethod.DENSE, split=EvaluationSplit.TEST)
    )
    assert run.status is EvaluationStatus.FAILED
    result = await service.get_result(run.run_id)
    assert result is not None
    assert result.error_message
    assert ("bozuk JSON" if invalid_data else "test sorusu yok") in result.error_message
    assert result.metrics is None
    assert (await service.list_results()).results == [result]


class MockRunner(UnusedRunner):
    is_mock = True


@pytest.mark.asyncio
async def test_service_does_not_persist_mock_metrics(database: SQLiteDatabase) -> None:
    await database.initialize()
    service = SQLiteEvaluationService(SQLiteEvaluationRepository(database), MockRunner())
    with pytest.raises(ValueError, match="Sahte"):
        await service.run_evaluation(REQUEST)
    assert (await service.list_results()).total == 0
