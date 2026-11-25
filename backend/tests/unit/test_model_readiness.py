import asyncio
import math
import threading
from dataclasses import FrozenInstanceError

import pytest

from backend.app.rag.model_readiness import (
    ModelReadinessManager,
    ReadinessState,
)


def noop() -> None:
    pass


def test_initial_snapshot() -> None:
    manager = ModelReadinessManager(
        embedding_warmup=noop,
        reranker_warmup=noop,
    )

    snapshot = manager.snapshot()

    assert snapshot.state == ReadinessState.NOT_STARTED
    assert snapshot.active_component is None
    assert snapshot.completed_components == ()
    assert snapshot.failed_component is None
    assert snapshot.component_durations_ms == ()
    assert snapshot.total_duration_ms is None


@pytest.mark.asyncio
async def test_successful_warmup_order_and_durations() -> None:
    calls = []

    def embedding() -> None:
        calls.append("embedding")

    def reranker() -> None:
        calls.append("reranker")

    async def llm() -> None:
        calls.append("llm")

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
        llm_warmup=llm,
    )

    await manager.warm_up()

    snapshot = manager.snapshot()
    expected = ("embedding", "reranker", "llm")

    assert calls == list(expected)
    assert snapshot.state == ReadinessState.READY
    assert snapshot.completed_components == expected
    assert snapshot.active_component is None
    assert snapshot.failed_component is None

    durations = dict(snapshot.component_durations_ms)

    assert tuple(durations) == expected
    assert all(
        math.isfinite(value) and value >= 0
        for value in durations.values()
    )

    assert snapshot.total_duration_ms is not None
    assert math.isfinite(snapshot.total_duration_ms)
    assert snapshot.total_duration_ms >= sum(durations.values())


@pytest.mark.asyncio
async def test_warmup_without_local_llm() -> None:
    manager = ModelReadinessManager(
        embedding_warmup=noop,
        reranker_warmup=noop,
        llm_warmup=None,
    )

    await manager.warm_up()

    snapshot = manager.snapshot()

    assert snapshot.state == ReadinessState.READY
    assert snapshot.completed_components == (
        "embedding",
        "reranker",
    )
    assert "llm" not in dict(snapshot.component_durations_ms)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_component", "expected_completed"),
    [
        ("embedding", ()),
        ("reranker", ("embedding",)),
        ("llm", ("embedding", "reranker")),
    ],
)
async def test_component_failure_stops_warmup(
    failed_component: str,
    expected_completed: tuple[str, ...],
) -> None:
    calls = []

    def run_component(name: str) -> None:
        calls.append(name)

        if name == failed_component:
            raise RuntimeError("Simulated model failure")

    async def llm() -> None:
        run_component("llm")

    manager = ModelReadinessManager(
        embedding_warmup=lambda: run_component("embedding"),
        reranker_warmup=lambda: run_component("reranker"),
        llm_warmup=llm,
    )

    await manager.warm_up()

    snapshot = manager.snapshot()
    expected_attempted = (*expected_completed, failed_component)

    assert snapshot.state == ReadinessState.FAILED
    assert snapshot.failed_component == failed_component
    assert snapshot.completed_components == expected_completed
    assert snapshot.active_component is None
    assert calls == list(expected_attempted)

    durations = dict(snapshot.component_durations_ms)

    assert tuple(durations) == expected_attempted
    assert all(
        math.isfinite(value) and value >= 0
        for value in durations.values()
    )
    assert snapshot.total_duration_ms is not None


@pytest.mark.asyncio
async def test_preparing_snapshot_and_concurrent_call() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    llm_calls = []

    async def llm() -> None:
        llm_calls.append("llm")
        entered.set()
        await release.wait()

    manager = ModelReadinessManager(
        embedding_warmup=noop,
        reranker_warmup=noop,
        llm_warmup=llm,
    )

    task = asyncio.create_task(manager.warm_up())

    try:
        await asyncio.wait_for(entered.wait(), timeout=5)

        snapshot = manager.snapshot()

        assert snapshot.state == ReadinessState.PREPARING
        assert snapshot.active_component == "llm"
        assert snapshot.completed_components == (
            "embedding",
            "reranker",
        )
        assert snapshot.failed_component is None
        assert snapshot.total_duration_ms is None

        await asyncio.wait_for(manager.warm_up(), timeout=5)

        assert manager.snapshot().state == ReadinessState.PREPARING
        assert llm_calls == ["llm"]

    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)

    assert manager.snapshot().state == ReadinessState.READY


@pytest.mark.asyncio
async def test_repeated_successful_warmup_is_noop() -> None:
    calls = []

    def embedding() -> None:
        calls.append("embedding")

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=noop,
    )

    await manager.warm_up()
    first_snapshot = manager.snapshot()

    await manager.warm_up()

    assert calls == ["embedding"]
    assert manager.snapshot() == first_snapshot


@pytest.mark.asyncio
async def test_failed_warmup_is_not_retried() -> None:
    calls = []

    def embedding() -> None:
        calls.append("embedding")
        raise RuntimeError("Simulated failure")

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=noop,
    )

    await manager.warm_up()
    first_snapshot = manager.snapshot()

    await manager.warm_up()

    assert calls == ["embedding"]
    assert first_snapshot.state == ReadinessState.FAILED
    assert manager.snapshot() == first_snapshot


@pytest.mark.asyncio
async def test_cancellation_updates_state_and_propagates() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def llm() -> None:
        entered.set()
        await release.wait()

    manager = ModelReadinessManager(
        embedding_warmup=noop,
        reranker_warmup=noop,
        llm_warmup=llm,
    )

    task = asyncio.create_task(manager.warm_up())

    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    snapshot = manager.snapshot()

    assert snapshot.state == ReadinessState.FAILED
    assert snapshot.failed_component == "llm"
    assert snapshot.active_component is None
    assert snapshot.completed_components == (
        "embedding",
        "reranker",
    )
    assert snapshot.total_duration_ms is not None
    assert "llm" in dict(snapshot.component_durations_ms)


def test_snapshot_is_immutable() -> None:
    manager = ModelReadinessManager(
        embedding_warmup=noop,
        reranker_warmup=noop,
    )

    snapshot = manager.snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.state = ReadinessState.READY


@pytest.mark.asyncio
async def test_old_snapshot_does_not_change() -> None:
    manager = ModelReadinessManager(
        embedding_warmup=noop,
        reranker_warmup=noop,
    )

    initial_snapshot = manager.snapshot()

    await manager.warm_up()

    assert initial_snapshot.state == ReadinessState.NOT_STARTED
    assert initial_snapshot.completed_components == ()
    assert initial_snapshot.component_durations_ms == ()
    assert manager.snapshot().state == ReadinessState.READY


@pytest.mark.asyncio
async def test_sync_warmups_run_in_worker_threads() -> None:
    event_loop_thread = threading.get_ident()
    thread_ids = {}

    def embedding() -> None:
        thread_ids["embedding"] = threading.get_ident()

    def reranker() -> None:
        thread_ids["reranker"] = threading.get_ident()

    async def llm() -> None:
        thread_ids["llm"] = threading.get_ident()

    manager = ModelReadinessManager(
        embedding_warmup=embedding,
        reranker_warmup=reranker,
        llm_warmup=llm,
    )

    await manager.warm_up()

    assert manager.snapshot().state == ReadinessState.READY
    assert thread_ids["embedding"] != event_loop_thread
    assert thread_ids["reranker"] != event_loop_thread
    assert thread_ids["llm"] == event_loop_thread
