from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from backend.app.schemas.evaluations import (
    EvaluationSplit,
    EvaluationStatus,
)
from backend.app.schemas.retrieval import RetrievalMethod
from scripts import run_evaluation


@pytest.mark.asyncio
async def test_cli_passes_real_runner_factory(monkeypatch):
    database = Mock()
    database.initialize = AsyncMock()

    store = Mock()
    repository = Mock()
    repository.list_case_results = AsyncMock(return_value=[])

    runner = Mock()
    runner_factory = Mock(return_value=runner)

    run_id = uuid4()
    result = Mock()
    result.status = EvaluationStatus.COMPLETED
    result.model_dump.return_value = {
        "status": "completed",
        "is_mock": False,
    }

    service = Mock()
    service.get_result = AsyncMock(return_value=result)

    async def execute_evaluation(request):
        factory = service_constructor.call_args.args[1]
        assert factory() is runner
        return SimpleNamespace(run_id=run_id)

    service.run_evaluation = AsyncMock(
        side_effect=execute_evaluation
    )

    service_constructor = Mock(return_value=service)
    database_constructor = Mock(return_value=database)
    repository_constructor = Mock(return_value=repository)
    store_factory = Mock(return_value=store)

    monkeypatch.setattr(
        run_evaluation, "SQLiteDatabase", database_constructor
    )
    monkeypatch.setattr(
        run_evaluation, "get_vector_store", store_factory
    )
    monkeypatch.setattr(
        run_evaluation,
        "SQLiteEvaluationRepository",
        repository_constructor,
    )
    monkeypatch.setattr(
        run_evaluation,
        "SQLiteEvaluationService",
        service_constructor,
    )
    monkeypatch.setattr(
        run_evaluation,
        "create_evaluation_runner",
        runner_factory,
    )

    exit_code = await run_evaluation.run(
        RetrievalMethod.HYBRID_RERANK,
        EvaluationSplit.TEST,
    )

    assert exit_code == 0

    database_constructor.assert_called_once()
    database.initialize.assert_awaited_once_with()
    store_factory.assert_called_once_with()
    repository_constructor.assert_called_once_with(database)

    service_constructor.assert_called_once()
    assert service_constructor.call_args.args[0] is repository
    assert callable(service_constructor.call_args.args[1])

    runner_factory.assert_called_once_with(store)

    service.run_evaluation.assert_awaited_once()
    request = service.run_evaluation.await_args.args[0]

    assert request.method is RetrievalMethod.HYBRID_RERANK
    assert request.split is EvaluationSplit.TEST

    service.get_result.assert_awaited_once_with(run_id)
    repository.list_case_results.assert_awaited_once_with(run_id)

    result.model_dump.assert_called_once_with(mode="json")
    store.store.close.assert_called_once_with()
