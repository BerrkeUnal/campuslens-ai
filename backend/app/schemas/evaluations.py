from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.schemas.retrieval import RetrievalMethod


class EvaluationSplit(StrEnum):
    VALIDATION = "validation"
    TEST = "test"


class EvaluationStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRunRequest(BaseModel):
    method: RetrievalMethod
    split: EvaluationSplit = EvaluationSplit.VALIDATION


class EvaluationMetricsResponse(BaseModel):
    total_cases: int = Field(ge=0)
    recall_at_5: float = Field(ge=0, le=1)
    mrr_at_5: float = Field(ge=0, le=1)
    answer_correctness: float = Field(ge=0, le=2)
    hallucination_rate: float = Field(ge=0, le=1)
    abstention_accuracy: float = Field(ge=0, le=1)
    average_latency_ms: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)


class EvaluationRunResponse(BaseModel):
    run_id: UUID
    method: RetrievalMethod
    split: EvaluationSplit
    status: EvaluationStatus
    is_mock: bool
    started_at: datetime


class EvaluationResultResponse(BaseModel):
    run_id: UUID
    method: RetrievalMethod
    split: EvaluationSplit
    status: EvaluationStatus
    metrics: EvaluationMetricsResponse | None
    is_mock: bool
    started_at: datetime
    completed_at: datetime | None


class EvaluationResultListResponse(BaseModel):
    results: list[EvaluationResultResponse]
    total: int = Field(ge=0)
