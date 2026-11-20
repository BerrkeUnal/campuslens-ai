from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from backend.app.schemas.evaluations import (
    EvaluationMetricsResponse,
    EvaluationResultListResponse,
    EvaluationResultResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationSplit,
    EvaluationStatus,
)
from backend.app.schemas.retrieval import RetrievalMethod


class EvaluationService(Protocol):
    async def run_evaluation(
        self,
        request: EvaluationRunRequest,
    ) -> EvaluationRunResponse: ...

    async def list_results(
        self,
    ) -> EvaluationResultListResponse: ...


class InMemoryEvaluationService:
    def __init__(self) -> None:
        self._results: dict[UUID, EvaluationResultResponse] = {}

    async def run_evaluation(
        self,
        request: EvaluationRunRequest,
    ) -> EvaluationRunResponse:
        run_id = uuid4()
        started_at = datetime.now(UTC)

        metrics = self._create_mock_metrics(
            method=request.method,
            split=request.split,
        )

        completed_at = datetime.now(UTC)

        result = EvaluationResultResponse(
            run_id=run_id,
            method=request.method,
            split=request.split,
            status=EvaluationStatus.COMPLETED,
            metrics=metrics,
            is_mock=True,
            started_at=started_at,
            completed_at=completed_at,
        )

        self._results[run_id] = result

        return EvaluationRunResponse(
            run_id=run_id,
            method=request.method,
            split=request.split,
            status=EvaluationStatus.COMPLETED,
            is_mock=True,
            started_at=started_at,
        )

    async def list_results(
        self,
    ) -> EvaluationResultListResponse:
        results = sorted(
            self._results.values(),
            key=lambda result: result.started_at,
            reverse=True,
        )

        return EvaluationResultListResponse(
            results=results,
            total=len(results),
        )

    def _create_mock_metrics(
        self,
        method: RetrievalMethod,
        split: EvaluationSplit,
    ) -> EvaluationMetricsResponse:
        total_cases = 10 if split == EvaluationSplit.VALIDATION else 40

        method_metrics = {
            RetrievalMethod.BM25: {
                "recall_at_5": 0.70,
                "mrr_at_5": 0.58,
                "answer_correctness": 1.20,
                "hallucination_rate": 0.15,
                "abstention_accuracy": 0.70,
                "average_latency_ms": 150,
                "total_cost_usd": 0.01,
            },
            RetrievalMethod.DENSE: {
                "recall_at_5": 0.78,
                "mrr_at_5": 0.65,
                "answer_correctness": 1.35,
                "hallucination_rate": 0.12,
                "abstention_accuracy": 0.75,
                "average_latency_ms": 190,
                "total_cost_usd": 0.01,
            },
            RetrievalMethod.HYBRID: {
                "recall_at_5": 0.84,
                "mrr_at_5": 0.72,
                "answer_correctness": 1.50,
                "hallucination_rate": 0.10,
                "abstention_accuracy": 0.80,
                "average_latency_ms": 230,
                "total_cost_usd": 0.02,
            },
            RetrievalMethod.HYBRID_RERANK: {
                "recall_at_5": 0.88,
                "mrr_at_5": 0.79,
                "answer_correctness": 1.65,
                "hallucination_rate": 0.08,
                "abstention_accuracy": 0.85,
                "average_latency_ms": 310,
                "total_cost_usd": 0.02,
            },
        }

        values = method_metrics[method]

        return EvaluationMetricsResponse(
            total_cases=total_cases,
            **values,
        )


_evaluation_service = InMemoryEvaluationService()


def get_evaluation_service() -> EvaluationService:
    return _evaluation_service
