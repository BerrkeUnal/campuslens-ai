"""Unit tests for the technical model readiness guard."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.app.core.exceptions import ApplicationError
from backend.app.rag.model_readiness import (
    ModelReadinessManager,
    ReadinessState,
)
from backend.app.rag.model_readiness_guard import (
    ModelPreparingError,
    ModelUnavailableError,
    ensure_models_ready,
)


def build_manager(state: ReadinessState | str) -> Mock:
    manager = Mock(spec=ModelReadinessManager)

    manager.snapshot.return_value = SimpleNamespace(
        state=state,
    )

    return manager


def test_disabled_warmup_allows_execution() -> None:
    assert ensure_models_ready(None) is None


@pytest.mark.parametrize(
    "state",
    [
        ReadinessState.NOT_STARTED,
        ReadinessState.PREPARING,
    ],
)
def test_unprepared_models_raise_preparing_error(
    state: ReadinessState,
) -> None:
    manager = build_manager(state)

    with pytest.raises(ModelPreparingError):
        ensure_models_ready(manager)

    manager.snapshot.assert_called_once_with()
    manager.warm_up.assert_not_called()


def test_ready_models_allow_execution() -> None:
    manager = build_manager(ReadinessState.READY)

    assert ensure_models_ready(manager) is None

    manager.snapshot.assert_called_once_with()
    manager.warm_up.assert_not_called()


def test_failed_models_raise_unavailable_error() -> None:
    manager = build_manager(ReadinessState.FAILED)

    with pytest.raises(ModelUnavailableError):
        ensure_models_ready(manager)

    manager.snapshot.assert_called_once_with()
    manager.warm_up.assert_not_called()


def test_unknown_state_fails_closed() -> None:
    manager = build_manager("unexpected-state")

    with pytest.raises(ModelUnavailableError):
        ensure_models_ready(manager)

    manager.snapshot.assert_called_once_with()
    manager.warm_up.assert_not_called()


def test_preparing_error_contract() -> None:
    error = ModelPreparingError()

    assert isinstance(error, ApplicationError)
    assert error.code == "model_preparing"
    assert error.status_code == 503

    assert error.message == (
        "Yapay zekâ modelleri hazırlanıyor. "
        "Lütfen kısa bir süre sonra tekrar deneyin."
    )

    assert str(error) == error.message


def test_unavailable_error_contract() -> None:
    error = ModelUnavailableError()

    assert isinstance(error, ApplicationError)
    assert error.code == "model_unavailable"
    assert error.status_code == 503

    assert error.message == (
        "Yapay zekâ modelleri şu anda kullanılamıyor. "
        "Lütfen daha sonra tekrar deneyin."
    )

    assert str(error) == error.message


def test_guard_does_not_change_readiness_snapshot() -> None:
    embedding = Mock()
    reranker = Mock()

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
    )

    before = manager.snapshot()

    with pytest.raises(ModelPreparingError):
        ensure_models_ready(manager)

    assert manager.snapshot() == before

    embedding.assert_not_called()
    reranker.assert_not_called()


def test_guard_does_not_expose_internal_failure_details() -> None:
    manager = build_manager(ReadinessState.FAILED)

    manager.snapshot.return_value.failed_component = (
        "private-model-cache-path"
    )

    with pytest.raises(ModelUnavailableError) as error:
        ensure_models_ready(manager)

    assert "private-model-cache-path" not in str(error.value)
    assert error.value.code == "model_unavailable"
