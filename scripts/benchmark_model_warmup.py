"""Benchmark real model warm-up with cold and warm residency."""

import argparse
import asyncio
import json
import os
import platform
import statistics
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from backend.app.core.config import get_settings
from backend.app.rag.embedder import get_embedder
from backend.app.rag.ollama_client import OllamaLLMClient
from backend.app.rag.reranker import get_reranker


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


async def measure(
    operation: Callable[[], Awaitable[None]],
) -> float:
    started = perf_counter()

    await operation()

    return (perf_counter() - started) * 1000


async def benchmark_component(
    name: str,
    operation: Callable[[], Awaitable[None]],
    repeats: int,
) -> dict:
    print(f"===== {name.upper()} =====", flush=True)

    cold_ms = await measure(operation)

    print(f"Cold: {cold_ms:.3f} ms", flush=True)

    warm_ms = []

    for index in range(repeats):
        duration = await measure(operation)

        warm_ms.append(duration)

        print(
            f"Warm {index + 1}: {duration:.3f} ms",
            flush=True,
        )

    summary = summarize(warm_ms)

    speedup = cold_ms / statistics.median(warm_ms)

    print(
        f"Warm median: {summary['median_ms']:.3f} ms",
        flush=True,
    )

    print(f"Speedup: {speedup:.3f}x", flush=True)

    return {
        "cold_ms": round(cold_ms, 3),
        "warm_ms": [round(value, 3) for value in warm_ms],
        "warm_summary": summary,
        "speedup_vs_warm_median": round(speedup, 3),
    }


async def run_benchmark(repeats: int, output: Path) -> None:
    settings = get_settings()

    if settings.llm_provider.strip().lower() != "ollama":
        raise ValueError("Benchmark requires the local Ollama provider.")

    if settings.ollama_model != "qwen3:4b":
        raise ValueError("Benchmark requires qwen3:4b.")

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("HF_HUB_OFFLINE=1 is required.")

    embedder = get_embedder()
    reranker = get_reranker()
    llm = OllamaLLMClient()

    async def embedding_operation() -> None:
        await asyncio.to_thread(embedder.warm_up)

    async def reranker_operation() -> None:
        await asyncio.to_thread(reranker.warm_up)

    try:
        results = {
            "embedding": await benchmark_component(
                "embedding",
                embedding_operation,
                repeats,
            ),
            "reranker": await benchmark_component(
                "reranker",
                reranker_operation,
                repeats,
            ),
            "llm": await benchmark_component(
                "llm",
                llm.warm_up,
                repeats,
            ),
        }

    finally:
        await llm.aclose()

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "mock": False,
        "warm_repeats": repeats,
        "offline_model_loading": True,
        "cold_definition": (
            "Fresh Python process and externally verified "
            "unloaded Ollama model; local disk cache retained."
        ),
        "models": {
            "dense": settings.embedding_model,
            "sparse": settings.sparse_model,
            "reranker": settings.reranker_model,
            "reranker_device": "cpu",
            "llm": settings.ollama_model,
            "ollama_keep_alive": settings.ollama_keep_alive,
        },
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Benchmark report: {output}", flush=True)
    print("BENCHMARK COMPLETED", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive.")

    if args.output.exists():
        parser.error("Output file already exists.")

    asyncio.run(run_benchmark(args.repeats, args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
