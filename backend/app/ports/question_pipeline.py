from typing import Protocol

from backend.app.schemas.questions import PipelineResult
from backend.app.schemas.retrieval import RetrievalMethod


class QuestionPipeline(Protocol):
    async def answer_question(
        self,
        question: str,
        method: RetrievalMethod,
    ) -> PipelineResult: ...
