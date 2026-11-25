from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from backend.app.rag.hybrid_rerank import search_hybrid_rerank
from backend.app.schemas.evaluations import (
    EvaluationSplit,
    EvaluationStatus,
)
from backend.app.schemas.retrieval import RetrievalMethod
from scripts import run_evaluation


@pytest.mark.asyncio
async def test_cli_passes_real_hybrid_rerank_adapter(monkeypatch):
    database = Mock()
    database.initialize = AsyncMock()

    store = Mock()
    repository = Mock()
    repository.list_case_results = AsyncMock(return_value=[])

    run_id = uuid4()
    result = Mock()
    result.status = EvaluationStatus.COMPLETED
    result.model_dump.return_value = {
        "status": "completed",
        "is_mock": False,
    }

    service = Mock()
    service.run_evaluation = AsyncMock(return_value=SimpleNamespace(run_id=run_id))
    service.get_result = AsyncMock(return_value=result)

    factory = Mock(return_value=service)

    monkeypatch.setattr(run_evaluation, "SQLiteDatabase", Mock(return_value=database))
    monkeypatch.setattr(run_evaluation, "get_vector_store", Mock(return_value=store))
    monkeypatch.setattr(
        run_evaluation,
        "SQLiteEvaluationRepository",
        Mock(return_value=repository),
    )
    monkeypatch.setattr(run_evaluation, "get_sqlite_evaluation_service", factory)

    exit_code = await run_evaluation.run(
        RetrievalMethod.HYBRID_RERANK,
        EvaluationSplit.VALIDATION,
    )

    assert exit_code == 0

    factory.assert_called_once_with(
        database,
        store,
        hybrid_rerank=search_hybrid_rerank,
    )

    request = service.run_evaluation.await_args.args[0]

    assert request.method is RetrievalMethod.HYBRID_RERANK
    assert request.split is EvaluationSplit.VALIDATION

    store.store.close.assert_called_once()
