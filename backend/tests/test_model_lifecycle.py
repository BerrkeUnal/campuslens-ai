"""Integration tests for application model readiness lifecycle."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.db.database import get_database
from backend.app.main import create_app
from backend.app.rag.model_readiness import (
    ModelReadinessManager,
    ReadinessState,
)
from backend.app.rag.model_settings import ModelSettingsMismatchError
from backend.app.rag.qdrant_store import get_vector_store

FACTORY = "backend.app.main.create_model_readiness_manager"
SETTINGS = "backend.app.main.get_settings"


@pytest.fixture
def runtime_settings() -> Settings:
    return Settings(
        _env_file=None,
        app_env="testing",
        database_url="sqlite:///./test.db",
        model_warmup_enabled=True,
        embedding_model="intfloat/multilingual-e5-small",
        sparse_model="Qdrant/bm25",
        reranker_model="BAAI/bge-reranker-v2-m3",
        llm_provider="ollama",
        llm_temperature=0.0,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="qwen3:4b",
        ollama_keep_alive="30m",
    )


@pytest.fixture
def readiness_manager() -> ModelReadinessManager:
    return ModelReadinessManager(
        embedding_warmup=Mock(),
        reranker_warmup=Mock(),
        llm_warmup=AsyncMock(),
    )


def test_disabled_warmup_does_not_create_manager(
    runtime_settings: Settings,
) -> None:
    settings = runtime_settings.model_copy(
        update={"model_warmup_enabled": False},
    )

    with patch(FACTORY) as factory:
        app = create_app(settings)

    factory.assert_not_called()
    assert app.state.model_readiness is None
    assert app.state.model_warmup_task is None


def test_disabled_warmup_ignores_injected_manager(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    settings = runtime_settings.model_copy(
        update={"model_warmup_enabled": False},
    )

    with patch(FACTORY) as factory:
        app = create_app(
            settings,
            readiness_manager=readiness_manager,
        )

    factory.assert_not_called()
    assert app.state.model_readiness is None
    assert app.state.model_warmup_task is None

    assert (
        readiness_manager.snapshot().state
        == ReadinessState.NOT_STARTED
    )


def test_enabled_warmup_creates_production_manager(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    with (
        patch(SETTINGS, return_value=runtime_settings),
        patch(FACTORY, return_value=readiness_manager) as factory,
    ):
        app = create_app(runtime_settings)

    factory.assert_called_once_with()

    assert app.state.model_readiness is readiness_manager
    assert app.state.model_warmup_task is None


def test_application_creation_does_not_run_warmup(
    runtime_settings: Settings,
) -> None:
    embedding = Mock()
    reranker = Mock()
    llm = AsyncMock()

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
        llm_warmup=llm,
    )

    with (
        patch(SETTINGS, return_value=runtime_settings),
        patch(FACTORY, return_value=manager),
    ):
        app = create_app(runtime_settings)

    embedding.assert_not_called()
    reranker.assert_not_called()
    llm.assert_not_called()
    llm.assert_not_awaited()

    assert manager.snapshot().state == ReadinessState.NOT_STARTED
    assert app.state.model_warmup_task is None


def test_model_configuration_mismatch_rejects_application(
    runtime_settings: Settings,
) -> None:
    application_settings = runtime_settings.model_copy(
        update={"embedding_model": "different-embedding"},
    )

    with (
        patch(SETTINGS, return_value=runtime_settings),
        patch(FACTORY) as factory,
        pytest.raises(
            ModelSettingsMismatchError,
            match="embedding_model",
        ),
    ):
        create_app(application_settings)

    factory.assert_not_called()


def test_non_model_application_overrides_are_allowed(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    application_settings = runtime_settings.model_copy(
        update={
            "app_env": "development",
            "frontend_origin": "http://localhost:8080",
            "database_url": "sqlite:///./different-test.db",
        },
    )

    with (
        patch(SETTINGS, return_value=runtime_settings),
        patch(FACTORY, return_value=readiness_manager) as factory,
    ):
        app = create_app(application_settings)

    factory.assert_called_once_with()

    assert app.state.model_readiness is readiness_manager
    assert app.state.model_warmup_task is None

    assert app.state.cors_allowed_origins == (
        "http://localhost:8080",
    )


def test_injected_manager_bypasses_production_factory(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    with patch(FACTORY) as factory:
        app = create_app(
            runtime_settings,
            readiness_manager=readiness_manager,
        )

    factory.assert_not_called()

    assert app.state.model_readiness is readiness_manager
    assert app.state.model_warmup_task is None


def test_injected_manager_bypasses_configuration_guard(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    application_settings = runtime_settings.model_copy(
        update={"embedding_model": "test-only-embedding"},
    )

    with (
        patch(SETTINGS, return_value=runtime_settings),
        patch(FACTORY) as factory,
        patch(
            "backend.app.main.validate_model_settings_compatibility",
        ) as guard,
    ):
        app = create_app(
            application_settings,
            readiness_manager=readiness_manager,
        )

    factory.assert_not_called()
    guard.assert_not_called()

    assert app.state.model_readiness is readiness_manager


def test_application_instances_have_independent_managers(
    runtime_settings: Settings,
) -> None:
    first_manager = ModelReadinessManager(
        embedding_warmup=Mock(),
        reranker_warmup=Mock(),
    )
    second_manager = ModelReadinessManager(
        embedding_warmup=Mock(),
        reranker_warmup=Mock(),
    )

    with (
        patch(SETTINGS, return_value=runtime_settings),
        patch(
            FACTORY,
            side_effect=[first_manager, second_manager],
        ) as factory,
    ):
        first_app = create_app(runtime_settings)
        second_app = create_app(runtime_settings)

    assert factory.call_count == 2

    assert first_app.state.model_readiness is first_manager
    assert second_app.state.model_readiness is second_manager

    assert (
        first_app.state.model_readiness
        is not second_app.state.model_readiness
    )

    assert first_app.state.model_warmup_task is None
    assert second_app.state.model_warmup_task is None


def test_production_factory_failure_is_not_suppressed(
    runtime_settings: Settings,
) -> None:
    with (
        patch(SETTINGS, return_value=runtime_settings),
        patch(
            FACTORY,
            side_effect=RuntimeError("test factory failure"),
        ) as factory,
        pytest.raises(
            RuntimeError,
            match="test factory failure",
        ),
    ):
        create_app(runtime_settings)

    factory.assert_called_once_with()


@pytest.mark.asyncio
async def test_disabled_lifespan_initializes_database(
    runtime_settings: Settings,
) -> None:
    settings = runtime_settings.model_copy(
        update={"model_warmup_enabled": False},
    )
    app = create_app(settings)

    with patch.object(
        app.state.database,
        "initialize",
        new_callable=AsyncMock,
    ) as initialize:
        async with app.router.lifespan_context(app):
            initialize.assert_awaited_once_with()

            assert app.state.model_readiness is None
            assert app.state.model_warmup_task is None

    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_database_initialization_precedes_model_warmup(
    runtime_settings: Settings,
) -> None:
    events: list[str] = []

    async def initialize() -> None:
        events.append("database")

    def embedding_warmup() -> None:
        events.append("embedding")

    def reranker_warmup() -> None:
        events.append("reranker")

    async def llm_warmup() -> None:
        events.append("llm")

    manager = ModelReadinessManager(
        embedding_warmup=embedding_warmup,
        reranker_warmup=reranker_warmup,
        llm_warmup=llm_warmup,
    )

    app = create_app(
        runtime_settings,
        readiness_manager=manager,
    )

    with patch.object(
        app.state.database,
        "initialize",
        side_effect=initialize,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert task is not None
            await asyncio.wait_for(task, timeout=5)

    assert events == [
        "database",
        "embedding",
        "reranker",
        "llm",
    ]

    assert manager.snapshot().state == ReadinessState.READY


@pytest.mark.asyncio
async def test_database_initialization_failure_prevents_warmup(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    app = create_app(
        runtime_settings,
        readiness_manager=readiness_manager,
    )

    with (
        patch.object(
            app.state.database,
            "initialize",
            new_callable=AsyncMock,
            side_effect=RuntimeError("test database failure"),
        ) as initialize,
        pytest.raises(
            RuntimeError,
            match="test database failure",
        ),
    ):
        async with app.router.lifespan_context(app):
            pytest.fail("Lifespan should not enter after DB failure.")

    initialize.assert_awaited_once_with()

    assert (
        readiness_manager.snapshot().state
        == ReadinessState.NOT_STARTED
    )

    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_startup_does_not_wait_for_model_warmup(
    runtime_settings: Settings,
) -> None:
    llm_started = asyncio.Event()
    release_llm = asyncio.Event()

    async def controlled_llm_warmup() -> None:
        llm_started.set()
        await release_llm.wait()

    manager = ModelReadinessManager(
        embedding_warmup=Mock(),
        reranker_warmup=Mock(),
        llm_warmup=controlled_llm_warmup,
    )

    app = create_app(
        runtime_settings,
        readiness_manager=manager,
    )

    async def exercise_lifespan() -> None:
        with patch.object(
            app.state.database,
            "initialize",
            new_callable=AsyncMock,
        ):
            async with app.router.lifespan_context(app):
                try:
                    await asyncio.wait_for(
                        llm_started.wait(),
                        timeout=5,
                    )

                    snapshot = manager.snapshot()
                    task = app.state.model_warmup_task

                    assert task is not None
                    assert not task.done()
                    assert snapshot.state == ReadinessState.PREPARING
                    assert snapshot.active_component == "llm"

                    assert snapshot.completed_components == (
                        "embedding",
                        "reranker",
                    )
                finally:
                    release_llm.set()

    try:
        await asyncio.wait_for(
            exercise_lifespan(),
            timeout=10,
        )
    finally:
        release_llm.set()

    assert manager.snapshot().state == ReadinessState.READY
    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_successful_background_warmup_reaches_ready(
    runtime_settings: Settings,
) -> None:
    embedding = Mock()
    reranker = Mock()
    llm = AsyncMock()

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
        llm_warmup=llm,
    )

    app = create_app(
        runtime_settings,
        readiness_manager=manager,
    )

    with patch.object(
        app.state.database,
        "initialize",
        new_callable=AsyncMock,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert task is not None
            await asyncio.wait_for(task, timeout=5)

            snapshot = manager.snapshot()

            assert snapshot.state == ReadinessState.READY
            assert snapshot.active_component is None
            assert snapshot.failed_component is None

            assert snapshot.completed_components == (
                "embedding",
                "reranker",
                "llm",
            )

            assert snapshot.total_duration_ms is not None
            assert snapshot.total_duration_ms >= 0

    embedding.assert_called_once_with()
    reranker.assert_called_once_with()
    llm.assert_awaited_once_with()

    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_failed_background_warmup_reaches_failed(
    runtime_settings: Settings,
) -> None:
    embedding = Mock()
    reranker = Mock(
        side_effect=RuntimeError("test reranker failure"),
    )
    llm = AsyncMock()

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
        llm_warmup=llm,
    )

    app = create_app(
        runtime_settings,
        readiness_manager=manager,
    )

    with patch.object(
        app.state.database,
        "initialize",
        new_callable=AsyncMock,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert task is not None
            await asyncio.wait_for(task, timeout=5)

            snapshot = manager.snapshot()

            assert snapshot.state == ReadinessState.FAILED
            assert snapshot.failed_component == "reranker"
            assert snapshot.active_component is None

            assert snapshot.completed_components == (
                "embedding",
            )

    embedding.assert_called_once_with()
    reranker.assert_called_once_with()
    llm.assert_not_called()

    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_background_task_is_tracked_in_application_state(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    app = create_app(
        runtime_settings,
        readiness_manager=readiness_manager,
    )

    assert app.state.model_warmup_task is None

    with patch.object(
        app.state.database,
        "initialize",
        new_callable=AsyncMock,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert isinstance(task, asyncio.Task)
            assert task.get_name() == "campuslens-model-warmup"

            await asyncio.wait_for(task, timeout=5)

            assert task.done()
            assert app.state.model_warmup_task is task

    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_openai_style_lifespan_skips_local_llm_warmup(
    runtime_settings: Settings,
) -> None:
    settings = runtime_settings.model_copy(
        update={"llm_provider": "openai"},
    )

    embedding = Mock()
    reranker = Mock()

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
        llm_warmup=None,
    )

    app = create_app(
        settings,
        readiness_manager=manager,
    )

    with patch.object(
        app.state.database,
        "initialize",
        new_callable=AsyncMock,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert task is not None
            await asyncio.wait_for(task, timeout=5)

            snapshot = manager.snapshot()

            assert snapshot.state == ReadinessState.READY

            assert snapshot.completed_components == (
                "embedding",
                "reranker",
            )

    embedding.assert_called_once_with()
    reranker.assert_called_once_with()

    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_shutdown_waits_for_running_warmup(
    runtime_settings: Settings,
) -> None:
    llm_started = asyncio.Event()
    release_llm = asyncio.Event()
    shutdown_started = asyncio.Event()
    shutdown_completed = asyncio.Event()

    async def controlled_llm_warmup() -> None:
        llm_started.set()
        await release_llm.wait()

    manager = ModelReadinessManager(
        embedding_warmup=Mock(),
        reranker_warmup=Mock(),
        llm_warmup=controlled_llm_warmup,
    )

    app = create_app(
        runtime_settings,
        readiness_manager=manager,
    )

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(
                llm_started.wait(),
                timeout=5,
            )

            shutdown_started.set()

        shutdown_completed.set()

    with patch.object(
        app.state.database,
        "initialize",
        new_callable=AsyncMock,
    ):
        lifecycle_task = asyncio.create_task(run_lifespan())

        try:
            await asyncio.wait_for(
                shutdown_started.wait(),
                timeout=5,
            )

            warmup_task = app.state.model_warmup_task

            assert warmup_task is not None
            assert not warmup_task.done()
            assert not lifecycle_task.done()
            assert not shutdown_completed.is_set()

            assert (
                manager.snapshot().state
                == ReadinessState.PREPARING
            )
        finally:
            release_llm.set()

            await asyncio.wait_for(
                lifecycle_task,
                timeout=5,
            )

    assert shutdown_completed.is_set()
    assert warmup_task.done()
    assert manager.snapshot().state == ReadinessState.READY
    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_shutdown_clears_completed_task_reference(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    app = create_app(
        runtime_settings,
        readiness_manager=readiness_manager,
    )

    with patch.object(
        app.state.database,
        "initialize",
        new_callable=AsyncMock,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert task is not None

            await asyncio.wait_for(task, timeout=5)

            assert task.done()
            assert task.exception() is None
            assert app.state.model_warmup_task is task

    assert app.state.model_warmup_task is None

    assert (
        readiness_manager.snapshot().state
        == ReadinessState.READY
    )


@pytest.mark.asyncio
async def test_shutdown_clears_task_after_unexpected_exception(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    app = create_app(
        runtime_settings,
        readiness_manager=readiness_manager,
    )

    unexpected_error = RuntimeError(
        "unexpected background task failure",
    )

    with (
        patch.object(
            app.state.database,
            "initialize",
            new_callable=AsyncMock,
        ),
        patch.object(
            readiness_manager,
            "warm_up",
            new_callable=AsyncMock,
            side_effect=unexpected_error,
        ) as warmup,
        pytest.raises(
            RuntimeError,
            match="unexpected background task failure",
        ) as exc_info,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert task is not None

    warmup.assert_awaited_once_with()

    assert exc_info.value is unexpected_error
    assert task.done()
    assert task.exception() is unexpected_error
    assert app.state.model_warmup_task is None


@pytest.mark.asyncio
async def test_shutdown_clears_cancelled_warmup_task(
    runtime_settings: Settings,
) -> None:
    llm_started = asyncio.Event()
    release_llm = asyncio.Event()

    async def controlled_llm_warmup() -> None:
        llm_started.set()
        await release_llm.wait()

    manager = ModelReadinessManager(
        embedding_warmup=Mock(),
        reranker_warmup=Mock(),
        llm_warmup=controlled_llm_warmup,
    )

    app = create_app(
        runtime_settings,
        readiness_manager=manager,
    )

    with (
        patch.object(
            app.state.database,
            "initialize",
            new_callable=AsyncMock,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        async with app.router.lifespan_context(app):
            try:
                await asyncio.wait_for(
                    llm_started.wait(),
                    timeout=5,
                )

                task = app.state.model_warmup_task

                assert task is not None
                assert task.cancel()
            finally:
                release_llm.set()

    assert task.cancelled()
    assert app.state.model_warmup_task is None

    snapshot = manager.snapshot()

    assert snapshot.state == ReadinessState.FAILED
    assert snapshot.failed_component == "llm"
    assert snapshot.active_component is None


@pytest.mark.asyncio
async def test_lifespan_body_exception_preserves_cleanup(
    runtime_settings: Settings,
    readiness_manager: ModelReadinessManager,
) -> None:
    app = create_app(
        runtime_settings,
        readiness_manager=readiness_manager,
    )

    body_error = RuntimeError("test lifespan body failure")

    with (
        patch.object(
            app.state.database,
            "initialize",
            new_callable=AsyncMock,
        ),
        pytest.raises(
            RuntimeError,
            match="test lifespan body failure",
        ) as exc_info,
    ):
        async with app.router.lifespan_context(app):
            task = app.state.model_warmup_task

            assert task is not None

            raise body_error

    assert exc_info.value is body_error
    assert task.done()
    assert task.exception() is None
    assert app.state.model_warmup_task is None

    assert (
        readiness_manager.snapshot().state
        == ReadinessState.READY
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "readiness_state",
        "database_status",
        "qdrant_status",
        "expected_status",
    ),
    [
        (
            ReadinessState.NOT_STARTED,
            "ok",
            "ok",
            "ok",
        ),
        (
            ReadinessState.PREPARING,
            "ok",
            "ok",
            "ok",
        ),
        (
            ReadinessState.READY,
            "ok",
            "ok",
            "ok",
        ),
        (
            ReadinessState.FAILED,
            "ok",
            "ok",
            "ok",
        ),
        (
            ReadinessState.PREPARING,
            "ok",
            "unavailable",
            "degraded",
        ),
        (
            ReadinessState.READY,
            "unavailable",
            "ok",
            "degraded",
        ),
        (
            ReadinessState.PREPARING,
            "ok",
            "collection_missing",
            "degraded",
        ),
        (
            ReadinessState.FAILED,
            "unavailable",
            "unavailable",
            "degraded",
        ),
    ],
    ids=[
        "not-started-healthy",
        "preparing-healthy",
        "ready-healthy",
        "failed-healthy",
        "preparing-qdrant-unavailable",
        "ready-database-unavailable",
        "preparing-collection-missing",
        "failed-both-unavailable",
    ],
)
async def test_health_contract_is_independent_of_model_readiness(
    runtime_settings: Settings,
    readiness_state: ReadinessState,
    database_status: str,
    qdrant_status: str,
    expected_status: str,
) -> None:
    llm_started = asyncio.Event()
    release_llm = asyncio.Event()

    async def controlled_llm_warmup() -> None:
        llm_started.set()

        if readiness_state == ReadinessState.FAILED:
            raise RuntimeError("test model warm-up failure")

        if readiness_state == ReadinessState.PREPARING:
            await release_llm.wait()

    embedding = Mock()
    reranker = Mock()

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
        llm_warmup=controlled_llm_warmup,
    )

    app = create_app(
        runtime_settings,
        readiness_manager=manager,
    )

    database = Mock()
    database.health_status = AsyncMock(
        return_value=database_status,
    )

    vector_store = Mock()
    vector_store.health_status = AsyncMock(
        return_value=qdrant_status,
    )

    app.dependency_overrides[get_database] = lambda: database
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    async def assert_health_response() -> None:
        before = manager.snapshot()

        assert before.state == readiness_state

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await asyncio.wait_for(
                client.get(
                    "/api/health",
                    headers={
                        "X-Request-ID": "model-health-test",
                    },
                ),
                timeout=5,
            )

        assert response.status_code == 200

        assert response.json() == {
            "status": expected_status,
            "backend": "ok",
            "database": database_status,
            "qdrant": qdrant_status,
        }

        assert (
            response.headers["X-Request-ID"]
            == "model-health-test"
        )

        assert manager.snapshot() == before

        database.health_status.assert_awaited_once_with()
        vector_store.health_status.assert_awaited_once_with()

    try:
        if readiness_state == ReadinessState.NOT_STARTED:
            await assert_health_response()

            embedding.assert_not_called()
            reranker.assert_not_called()
            assert not llm_started.is_set()
            assert app.state.model_warmup_task is None

        else:
            with patch.object(
                app.state.database,
                "initialize",
                new_callable=AsyncMock,
            ) as initialize:
                async with app.router.lifespan_context(app):
                    try:
                        initialize.assert_awaited_once_with()

                        task = app.state.model_warmup_task

                        assert task is not None

                        if readiness_state == ReadinessState.PREPARING:
                            await asyncio.wait_for(
                                llm_started.wait(),
                                timeout=5,
                            )

                            assert not task.done()

                        else:
                            await asyncio.wait_for(
                                task,
                                timeout=5,
                            )

                        await assert_health_response()

                        if readiness_state == ReadinessState.PREPARING:
                            assert not task.done()
                            assert not release_llm.is_set()

                    finally:
                        release_llm.set()

            assert app.state.model_warmup_task is None

            if readiness_state == ReadinessState.FAILED:
                assert (
                    manager.snapshot().state
                    == ReadinessState.FAILED
                )
            else:
                assert (
                    manager.snapshot().state
                    == ReadinessState.READY
                )

    finally:
        release_llm.set()
        app.dependency_overrides.clear()
