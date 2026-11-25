"""Coordinate model warm-up and expose technical readiness state."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

logger = logging.getLogger(__name__)

SyncWarmup = Callable[[], None]
AsyncWarmup = Callable[[], Awaitable[None]]


class ReadinessState(StrEnum):
    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    state: ReadinessState
    active_component: str | None
    completed_components: tuple[str, ...]
    failed_component: str | None
    component_durations_ms: tuple[tuple[str, float], ...]
    total_duration_ms: float | None


class ModelReadinessManager:
    """Prepare required models once and track their readiness."""

    def __init__(
        self,
        *,
        embedding_warmup: SyncWarmup,
        reranker_warmup: SyncWarmup,
        llm_warmup: AsyncWarmup | None = None,
    ) -> None:
        self._embedding_warmup = embedding_warmup
        self._reranker_warmup = reranker_warmup
        self._llm_warmup = llm_warmup

        self._state = ReadinessState.NOT_STARTED
        self._active_component: str | None = None
        self._completed_components: list[str] = []
        self._failed_component: str | None = None
        self._component_durations_ms: dict[str, float] = {}
        self._total_duration_ms: float | None = None

    def snapshot(self) -> ReadinessSnapshot:
        """Return the current state without waiting for model loading."""
        return ReadinessSnapshot(
            state=self._state,
            active_component=self._active_component,
            completed_components=tuple(self._completed_components),
            failed_component=self._failed_component,
            component_durations_ms=tuple(
                self._component_durations_ms.items()
            ),
            total_duration_ms=self._total_duration_ms,
        )

    async def _run_component(
        self,
        name: str,
        warmup: AsyncWarmup,
    ) -> None:
        self._active_component = name
        started = perf_counter()

        logger.info("Model warm-up started: %s", name)

        try:
            await warmup()
        finally:
            self._component_durations_ms[name] = (
                perf_counter() - started
            ) * 1000

        self._completed_components.append(name)
        self._active_component = None

        logger.info(
            "Model warm-up completed: %s (%.2f ms)",
            name,
            self._component_durations_ms[name],
        )

    async def warm_up(self) -> None:
        """Run the configured warm-up steps sequentially."""
        if self._state != ReadinessState.NOT_STARTED:
            return

        self._state = ReadinessState.PREPARING
        started = perf_counter()

        try:
            await self._run_component(
                "embedding",
                lambda: asyncio.to_thread(self._embedding_warmup),
            )

            await self._run_component(
                "reranker",
                lambda: asyncio.to_thread(self._reranker_warmup),
            )

            if self._llm_warmup is not None:
                await self._run_component(
                    "llm",
                    self._llm_warmup,
                )

        except asyncio.CancelledError:
            self._failed_component = self._active_component
            self._state = ReadinessState.FAILED
            logger.warning("Model warm-up cancelled.")
            raise

        except Exception:
            self._failed_component = self._active_component
            self._state = ReadinessState.FAILED

            logger.exception(
                "Model warm-up failed: %s",
                self._failed_component,
            )

        else:
            self._state = ReadinessState.READY
            logger.info("All required model warm-ups completed.")

        finally:
            self._active_component = None
            self._total_duration_ms = (
                perf_counter() - started
            ) * 1000
