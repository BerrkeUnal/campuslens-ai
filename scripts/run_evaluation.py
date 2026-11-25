"""Run retrieval evaluation: python -m scripts.run_evaluation --method dense."""

import argparse
import asyncio
import json
from dataclasses import asdict

from backend.app.core.config import get_settings
from backend.app.db.database import SQLiteDatabase
from backend.app.db.evaluation_repository import SQLiteEvaluationRepository
from backend.app.rag.qdrant_store import get_vector_store
from backend.app.schemas.evaluations import (
    EvaluationRunRequest,
    EvaluationSplit,
    EvaluationStatus,
)
from backend.app.schemas.retrieval import RetrievalMethod
from backend.app.services.sqlite_evaluation_service import (
    SQLiteEvaluationService,
    create_evaluation_runner,
)


async def run(method: RetrievalMethod, split: EvaluationSplit) -> int:
    database = SQLiteDatabase(get_settings().database_url)
    await database.initialize()
    store = get_vector_store()
    try:
        repository = SQLiteEvaluationRepository(database)
        service = SQLiteEvaluationService(repository, lambda: create_evaluation_runner(store))
        run_result = await service.run_evaluation(EvaluationRunRequest(method=method, split=split))
        result = await service.get_result(run_result.run_id)
        assert result is not None
        cases = await repository.list_case_results(run_result.run_id)
        output = result.model_dump(mode="json")
        output["cases"] = [
            {
                **asdict(case),
                "source_hit": case.source_hit if case.answerable else None,
                "reciprocal_rank": (
                    case.reciprocal_rank if case.answerable else None
                ),
            }
            for case in cases
        ]
        print(json.dumps(output, ensure_ascii=True, indent=2))
        return 0 if result.status is EvaluationStatus.COMPLETED else 1
    finally:
        await asyncio.to_thread(store.store.close)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=RetrievalMethod, choices=list(RetrievalMethod), required=True)
    parser.add_argument(
        "--split", type=EvaluationSplit, choices=list(EvaluationSplit),
        default=EvaluationSplit.TEST,
    )
    args = parser.parse_args()
    return asyncio.run(run(args.method, args.split))


if __name__ == "__main__":
    raise SystemExit(main())
