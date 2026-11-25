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
    service = SQLiteEvaluationService(SQLiteEvaluationRepository(database), UnusedRunner, **options)
    run = await service.run_evaluation(
        EvaluationRunRequest(method=RetrievalMethod.DENSE, split=EvaluationSplit.VALIDATION)
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
    service = SQLiteEvaluationService(SQLiteEvaluationRepository(database), MockRunner)
    run = await service.run_evaluation(REQUEST)
    assert run.status == EvaluationStatus.FAILED
    result = await service.get_result(run.run_id)
    assert result is not None and result.metrics is None
    assert await SQLiteEvaluationRepository(database).list_case_results(run.run_id) == []


@pytest.mark.asyncio
async def test_unanswerable_results_are_saved_and_legacy_mock_rows_are_hidden(
    database: SQLiteDatabase,
) -> None:
    await database.initialize()
    repository = SQLiteEvaluationRepository(database)
    cases = [
        RetrievalCaseResult("answerable", 1, 10),
        RetrievalCaseResult("abstained", None, 15, True, "Kaynaklarda bilgi yok."),
        RetrievalCaseResult("answered", None, 20, False, "Unsupported answer."),
        RetrievalCaseResult("unmeasured", None, 25, answer_error="Unavailable"),
    ]
    run = await repository.start_run(REQUEST, total_cases=len(cases))
    await repository.complete_run(run.run_id, RetrievalEvaluation(cases, summarize_retrieval(cases)))
    hidden = await repository.start_run(REQUEST, total_cases=0)
    async with database.connect() as connection:
        await connection.execute(
            "UPDATE evaluation_runs SET is_mock = 1 WHERE run_id = ?", (str(hidden.run_id),)
        )
        await connection.commit()

    result, = await repository.list_results()
    assert result.metrics is not None
    assert result.metrics.recall_at_5 == 1
    assert result.metrics.abstention_accuracy == 0.5
    assert result.metrics.abstention_total == 3
    assert result.metrics.abstention_evaluated == 2
    assert await repository.list_case_results(run.run_id) == cases
    assert await repository.get_result(hidden.run_id) is None


def test_create_evaluation_runner_wires_production_components(
    monkeypatch,
) -> None:
    from unittest.mock import Mock

    from backend.app.services import sqlite_evaluation_service as module

    store = Mock()
    retriever = Mock()
    reranker = Mock()
    reranked_search = Mock()
    pipeline = Mock()
    runner = Mock()

    retriever_constructor = Mock(return_value=retriever)
    reranker_provider = Mock(return_value=reranker)
    hybrid_constructor = Mock(return_value=reranked_search)
    pipeline_constructor = Mock(return_value=pipeline)
    runner_constructor = Mock(return_value=runner)

    monkeypatch.setattr(module, "Retriever", retriever_constructor)
    monkeypatch.setattr(module, "get_reranker", reranker_provider)
    monkeypatch.setattr(module, "HybridRerankSearch", hybrid_constructor)
    monkeypatch.setattr(module, "RAGPipeline", pipeline_constructor)
    monkeypatch.setattr(
        module, "RetrievalEvaluationRunner", runner_constructor
    )

    result = module.create_evaluation_runner(store)

    assert result is runner

    retriever_constructor.assert_called_once_with(store)
    reranker_provider.assert_called_once_with()
    hybrid_constructor.assert_called_once_with(retriever, reranker)
    pipeline_constructor.assert_called_once_with(
        retriever, reranked_search
    )

    runner_constructor.assert_called_once_with(
        retriever,
        reranked_search.search,
        is_mock=False,
        answer_with_context=pipeline.answer_with_context,
    )
