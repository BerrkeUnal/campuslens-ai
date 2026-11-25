import logging
import math
from threading import Lock

from sentence_transformers import CrossEncoder

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ApplicationError
from backend.app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankerModelError(ApplicationError):
    """Raised when the reranker model is unavailable (HTTP 503)."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            code="reranker_unavailable",
            message=(
                "Yeniden sıralama modeli şu anda kullanılamıyor. "
                "Lütfen daha sonra tekrar deneyin."
            ),
            status_code=503,
        )
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


class Reranker:
    """Rerank retrieved candidates using a BGE cross-encoder."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().reranker_model
        self._model: CrossEncoder | None = None
        self._lock = Lock()

    def _get_model_locked(self) -> CrossEncoder:
        """Return the shared model while the caller holds the lock."""
        if self._model is None:
            try:
                self._model = CrossEncoder(
                    self._model_name,
                    device="cpu",
                    local_files_only=True,
                )
            except Exception as exc:
                logger.exception("Reranker model loading failed.")
                raise RerankerModelError(
                    f"Could not load reranker {self._model_name}: {exc}"
                ) from exc

        return self._model

    def warm_up(self) -> None:
        """Load the shared model and run a small inference request."""
        with self._lock:
            model = self._get_model_locked()

            try:
                scores = model.predict(
                    [
                        (
                            "Öğrenci kayıt işlemleri",
                            "Öğrenci kayıtları akademik takvime göre yapılır.",
                        )
                    ],
                    show_progress_bar=False,
                )

                if len(scores) != 1 or not math.isfinite(float(scores[0])):
                    raise ValueError("Invalid reranker warm-up score.")

            except Exception as exc:
                logger.exception("Reranker warm-up failed.")
                raise RerankerModelError(f"Reranker warm-up failed: {exc}") from exc

        logger.info("Reranker warm-up completed.")

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
            model = self._get_model_locked()

            try:
                scores = model.predict(
                    pairs,
                    show_progress_bar=False,
                )
            except Exception as exc:
                logger.exception("Reranker inference failed.")
                raise RerankerModelError(f"Reranker inference failed: {exc}") from exc

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


def rerank(
    question: str,
    candidates: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Rerank candidates using the shared reranker instance."""
    return get_reranker().rerank(
        question,
        candidates,
        top_k,
    )
