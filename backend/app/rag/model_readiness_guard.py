"""Guard question execution using technical model readiness."""

from backend.app.core.exceptions import ApplicationError
from backend.app.rag.model_readiness import (
    ModelReadinessManager,
    ReadinessState,
)


class ModelPreparingError(ApplicationError):
    """Required models have not completed startup warm-up."""

    def __init__(self) -> None:
        super().__init__(
            code="model_preparing",
            message=(
                "Yapay zekâ modelleri hazırlanıyor. "
                "Lütfen kısa bir süre sonra tekrar deneyin."
            ),
            status_code=503,
        )


class ModelUnavailableError(ApplicationError):
    """Required model warm-up failed."""

    def __init__(self) -> None:
        super().__init__(
            code="model_unavailable",
            message=(
                "Yapay zekâ modelleri şu anda kullanılamıyor. "
                "Lütfen daha sonra tekrar deneyin."
            ),
            status_code=503,
        )


def ensure_models_ready(
    manager: ModelReadinessManager | None,
) -> None:
    """Reject question execution until required models are ready.

    A missing manager means startup warm-up is disabled.
    This check never loads models or waits for warm-up.
    """
    if manager is None:
        return

    state = manager.snapshot().state

    if state == ReadinessState.READY:
        return

    if state in (
        ReadinessState.NOT_STARTED,
        ReadinessState.PREPARING,
    ):
        raise ModelPreparingError()

    raise ModelUnavailableError()
