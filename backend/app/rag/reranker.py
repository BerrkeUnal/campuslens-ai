import math
from threading import Lock

from sentence_transformers import CrossEncoder

from backend.app.core.config import get_settings
from backend.app.schemas.retrieval import RetrievedChunk


class Reranker:
    """Rerank retrieved candidates using a BGE cross-encoder."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().reranker_model
        self._model: CrossEncoder | None = None
        self._lock = Lock()

    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Score candidates and return the highest-ranked results."""
        if not question.strip():
            raise ValueError("Question must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        if not candidates:
            return []

        pairs = [(question, candidate.text) for candidate in candidates]

        with self._lock:
            if self._model is None:
                self._model = CrossEncoder(self._model_name)

            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
            )

        if len(scores) != len(candidates):
            raise ValueError("Reranker returned an unexpected score count.")

        ranked_candidates = []

        for candidate, score in zip(candidates, scores, strict=True):
            rerank_score = float(score)

            if not math.isfinite(rerank_score):
                raise ValueError("Reranker returned a non-finite score.")

            ranked_candidates.append(
                candidate.model_copy(
                    update={"rerank_score": rerank_score},
                )
            )

        ranked_candidates.sort(
            key=lambda candidate: candidate.rerank_score,
            reverse=True,
        )

        return ranked_candidates[:top_k]


_default_reranker: Reranker | None = None
_default_lock = Lock()


def get_reranker() -> Reranker:
    """Return the process-wide reranker instance."""
    global _default_reranker

    with _default_lock:
        if _default_reranker is None:
            _default_reranker = Reranker()

        return _default_reranker
