from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, StringConstraints

from backend.app.schemas.retrieval import RetrievalMethod, SourceResponse

QuestionText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=1000,
    ),
]


class AskRequest(BaseModel):
    question: QuestionText
    method: RetrievalMethod = RetrievalMethod.HYBRID_RERANK


class MetricsResponse(BaseModel):
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    estimated_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class PipelineResult(BaseModel):
    answer: str
    insufficient_context: bool
    sources: list[SourceResponse]
    metrics: MetricsResponse


class AskResponse(BaseModel):
    request_id: UUID
    answer: str
    insufficient_context: bool
    method: RetrievalMethod
    sources: list[SourceResponse]
    metrics: MetricsResponse
