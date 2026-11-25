"""Prepare required models before starting CampusLens."""

import asyncio
from time import perf_counter

import httpx
from sentence_transformers import CrossEncoder

from backend.app.core.config import get_settings
from backend.app.rag.embedder import TextEmbedder


def prepare_embeddings() -> None:
    start = perf_counter()

    print("[1/3] Preparing embedding models...", flush=True)

    settings = get_settings()

    embedder = TextEmbedder(
        settings.embedding_model,
        settings.sparse_model,
        local_files_only=False,
    )
    embedder.load()

    print(
        f"Embedding models prepared in {perf_counter() - start:.2f}s",
        flush=True,
    )


def prepare_reranker() -> None:
    settings = get_settings()
    start = perf_counter()

    print(
        f"[2/3] Preparing reranker: {settings.reranker_model}",
        flush=True,
    )

    CrossEncoder(
        settings.reranker_model,
        device="cpu",
        local_files_only=False,
    )

    print(
        f"Reranker prepared in {perf_counter() - start:.2f}s",
        flush=True,
    )


async def prepare_ollama() -> None:
    settings = get_settings()

    if settings.llm_provider.lower() != "ollama":
        print("[3/3] Ollama preparation skipped.")
        return

    start = perf_counter()
    model = settings.ollama_model
    base_url = settings.ollama_base_url.rstrip("/")

    print(f"[3/3] Checking Ollama model: {model}", flush=True)

    async with httpx.AsyncClient(timeout=1800.0) as client:
        response = await client.post(
            f"{base_url}/api/show",
            json={"model": model},
        )

        if response.status_code == 404:
            print(f"Downloading Ollama model: {model}", flush=True)

            response = await client.post(
                f"{base_url}/api/pull",
                json={
                    "model": model,
                    "stream": False,
                },
            )
            response.raise_for_status()

            response = await client.post(
                f"{base_url}/api/show",
                json={"model": model},
            )

        response.raise_for_status()

    print(
        f"Ollama model prepared in {perf_counter() - start:.2f}s",
        flush=True,
    )


async def prepare_models() -> None:
    start = perf_counter()

    await asyncio.to_thread(prepare_embeddings)
    await asyncio.to_thread(prepare_reranker)
    await prepare_ollama()

    print(
        f"All required models prepared in {perf_counter() - start:.2f}s",
        flush=True,
    )


def main() -> int:
    try:
        asyncio.run(prepare_models())
    # CLI boundary: report preparation failures with a nonzero exit code.
    except Exception as exc:  # noqa: BLE001
        print(
            f"Model preparation failed ({type(exc).__name__}): {exc}",
            flush=True,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
