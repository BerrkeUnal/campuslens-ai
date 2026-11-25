from datetime import UTC, datetime
from uuid import UUID, uuid4

from aiosqlite import Row

from backend.app.db.database import SQLiteDatabase
from backend.app.schemas.evaluations import (
    EvaluationMetricsResponse,
    EvaluationResultResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationStatus,
)
from backend.app.services.evaluation_metrics import RetrievalCaseResult
from backend.app.services.retrieval_evaluation import RetrievalEvaluation

RUN_RESULTS = """SELECT runs.*,
       (SELECT AVG(a.insufficient_context) FROM evaluation_abstentions AS a
        WHERE a.run_id = runs.run_id) AS abstention_accuracy,
       (SELECT COUNT(*) FROM evaluation_abstentions AS a
        WHERE a.run_id = runs.run_id) AS abstention_total,
       (SELECT COUNT(a.insufficient_context) FROM evaluation_abstentions AS a
        WHERE a.run_id = runs.run_id) AS abstention_evaluated
FROM evaluation_runs AS runs WHERE runs.is_mock = 0"""


def to_result(row: Row) -> EvaluationResultResponse:
    metrics = None
    if row["status"] == EvaluationStatus.COMPLETED:
        metrics = EvaluationMetricsResponse(
            total_cases=row["total_cases"],
            recall_at_5=row["recall_at_5"],
            mrr_at_5=row["mrr_at_5"],
            average_latency_ms=row["average_latency_ms"],
            abstention_accuracy=row["abstention_accuracy"],
            abstention_total=row["abstention_total"],
            abstention_evaluated=row["abstention_evaluated"],
        )
    return EvaluationResultResponse.model_validate({**dict(row), "metrics": metrics})


class SQLiteEvaluationRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def start_run(
        self, request: EvaluationRunRequest, total_cases: int
    ) -> EvaluationRunResponse:
        run = EvaluationRunResponse(
            run_id=uuid4(),
            method=request.method,
            split=request.split,
            status=EvaluationStatus.RUNNING,
            is_mock=False,
            started_at=datetime.now(UTC),
        )
        async with self._database.connect() as connection:
            await connection.execute(
                """INSERT INTO evaluation_runs
                   (run_id, method, split, status, total_cases, started_at, is_mock)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(run.run_id),
                    run.method.value,
                    run.split.value,
                    run.status.value,
                    total_cases,
                    run.started_at.isoformat(),
                    run.is_mock,
                ),
            )
            await connection.commit()
        return run

    async def complete_run(self, run_id: UUID, evaluation: RetrievalEvaluation) -> None:
        metrics = evaluation.metrics
        async with self._database.connect() as connection:
            async with connection.execute(
                """UPDATE evaluation_runs
                   SET status = 'completed', recall_at_5 = ?, mrr_at_5 = ?,
                       average_latency_ms = ?, completed_at = ?
                   WHERE run_id = ? AND status = 'running' AND total_cases = ?""",
                (
                    metrics.recall_at_5,
                    metrics.mrr_at_5,
                    metrics.average_latency_ms,
                    datetime.now(UTC).isoformat(),
                    str(run_id),
                    len(evaluation.cases),
                ),
            ) as cursor:
                if cursor.rowcount != 1:
                    raise ValueError("Değerlendirme koşusu veya soru sayısı geçersiz.")
            await connection.executemany(
                """INSERT INTO evaluation_case_results
                   (run_id, case_id, source_rank, source_hit, latency_ms)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (str(run_id), case.case_id, case.source_rank, case.source_hit, case.latency_ms)
                    for case in evaluation.cases
                ],
            )
            await connection.executemany(
                """INSERT INTO evaluation_abstentions
                   (run_id, case_id, insufficient_context, answer, error_message) VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        str(run_id), case.case_id, case.insufficient_context,
                        case.answer, case.answer_error,
                    )
                    for case in evaluation.cases
                    if not case.answerable
                ],
            )
            await connection.commit()

    async def fail_run(self, run_id: UUID, message: str) -> None:
        async with self._database.connect() as connection:
            await connection.execute(
                """UPDATE evaluation_runs SET status = 'failed', completed_at = ?, error_message = ?
                   WHERE run_id = ? AND status = 'running'""",
                (datetime.now(UTC).isoformat(), message, str(run_id)),
            )
            await connection.commit()

    async def get_result(self, run_id: UUID) -> EvaluationResultResponse | None:
        async with self._database.connect() as connection, connection.execute(
            f"{RUN_RESULTS} AND runs.run_id = ?", (str(run_id),)
        ) as cursor:
            row = await cursor.fetchone()
        return to_result(row) if row is not None else None

    async def list_results(self) -> list[EvaluationResultResponse]:
        async with self._database.connect() as connection:
            rows = await connection.execute_fetchall(
                f"{RUN_RESULTS} ORDER BY started_at DESC, run_id"
            )
        return [to_result(row) for row in rows]

    async def list_case_results(self, run_id: UUID) -> list[RetrievalCaseResult]:
        async with self._database.connect() as connection:
            rows = await connection.execute_fetchall(
                """SELECT r.case_id, r.source_rank, r.latency_ms, a.insufficient_context, a.answer, a.error_message
                   FROM evaluation_case_results AS r LEFT JOIN evaluation_abstentions AS a
                       ON a.run_id = r.run_id AND a.case_id = r.case_id
                   WHERE r.run_id = ? ORDER BY r.rowid""",
                (str(run_id),),
            )
        return [
            RetrievalCaseResult(
                row["case_id"], row["source_rank"], row["latency_ms"],
                None if row["insufficient_context"] is None else bool(row["insufficient_context"]),
                row["answer"],
                row["error_message"],
            )
            for row in rows
        ]
