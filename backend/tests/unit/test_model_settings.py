"""Unit tests for shared model settings compatibility."""

import pytest

from backend.app.core.config import Settings
from backend.app.rag.model_settings import (
    ModelSettingsMismatchError,
    validate_model_settings_compatibility,
)


@pytest.fixture
def runtime_settings() -> Settings:
    return Settings(
        _env_file=None,
        embedding_model="intfloat/multilingual-e5-small",
        sparse_model="Qdrant/bm25",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_device="cpu",
        llm_provider="ollama",
        llm_temperature=0.0,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="qwen3:4b",
        ollama_keep_alive="30m",
        llm_model="gpt-4.1-mini",
        openai_api_key="test-only-runtime-key",
    )


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_identical_settings_are_compatible(
    runtime_settings: Settings,
    provider: str,
) -> None:
    runtime = runtime_settings.model_copy(
        update={"llm_provider": provider},
    )
    application = runtime.model_copy()

    validate_model_settings_compatibility(application, runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_model", "different-embedding"),
        ("sparse_model", "different-sparse"),
        ("reranker_model", "different-reranker"),
        ("llm_temperature", 0.7),
    ],
)
def test_common_model_field_mismatch(
    runtime_settings: Settings,
    field: str,
    value: str | float,
) -> None:
    application = runtime_settings.model_copy(
        update={field: value},
    )

    with pytest.raises(
        ModelSettingsMismatchError,
        match=field,
    ):
        validate_model_settings_compatibility(
            application,
            runtime_settings,
        )


@pytest.mark.parametrize(
    ("runtime_provider", "application_provider"),
    [
        ("ollama", "openai"),
        ("openai", "ollama"),
    ],
)
def test_provider_mismatch(
    runtime_settings: Settings,
    runtime_provider: str,
    application_provider: str,
) -> None:
    runtime = runtime_settings.model_copy(
        update={"llm_provider": runtime_provider},
    )
    application = runtime.model_copy(
        update={"llm_provider": application_provider},
    )

    with pytest.raises(
        ModelSettingsMismatchError,
        match="llm_provider",
    ):
        validate_model_settings_compatibility(
            application,
            runtime,
        )


@pytest.mark.parametrize(
    ("provider", "normalized_variant"),
    [
        ("ollama", " OLLAMA "),
        ("openai", " OPENAI "),
    ],
)
def test_provider_normalization(
    runtime_settings: Settings,
    provider: str,
    normalized_variant: str,
) -> None:
    runtime = runtime_settings.model_copy(
        update={"llm_provider": provider},
    )
    application = runtime.model_copy(
        update={"llm_provider": normalized_variant},
    )

    validate_model_settings_compatibility(
        application,
        runtime,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ollama_base_url", "http://localhost:9999"),
        ("ollama_model", "llama3.1:8b"),
        ("ollama_keep_alive", "5m"),
    ],
)
def test_ollama_specific_mismatch(
    runtime_settings: Settings,
    field: str,
    value: str,
) -> None:
    application = runtime_settings.model_copy(
        update={field: value},
    )

    with pytest.raises(
        ModelSettingsMismatchError,
        match=field,
    ):
        validate_model_settings_compatibility(
            application,
            runtime_settings,
        )


def test_ollama_url_trailing_slash_normalization(
    runtime_settings: Settings,
) -> None:
    application = runtime_settings.model_copy(
        update={
            "ollama_base_url": (
                runtime_settings.ollama_base_url + "/"
            ),
        },
    )

    validate_model_settings_compatibility(
        application,
        runtime_settings,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_model", "different-openai-model"),
        ("openai_api_key", "different-test-only-key"),
    ],
)
def test_openai_specific_mismatch(
    runtime_settings: Settings,
    field: str,
    value: str,
) -> None:
    runtime = runtime_settings.model_copy(
        update={"llm_provider": "openai"},
    )
    application = runtime.model_copy(
        update={field: value},
    )

    with pytest.raises(
        ModelSettingsMismatchError,
        match=field,
    ):
        validate_model_settings_compatibility(
            application,
            runtime,
        )


def test_ollama_ignores_inactive_openai_settings(
    runtime_settings: Settings,
) -> None:
    application = runtime_settings.model_copy(
        update={
            "llm_model": "unused-openai-model",
            "openai_api_key": "unused-test-only-key",
        },
    )

    validate_model_settings_compatibility(
        application,
        runtime_settings,
    )


def test_openai_ignores_inactive_ollama_settings(
    runtime_settings: Settings,
) -> None:
    runtime = runtime_settings.model_copy(
        update={"llm_provider": "openai"},
    )
    application = runtime.model_copy(
        update={
            "ollama_base_url": "http://localhost:9999",
            "ollama_model": "unused-ollama-model",
            "ollama_keep_alive": "5m",
        },
    )

    validate_model_settings_compatibility(
        application,
        runtime,
    )


def test_non_model_application_overrides_are_allowed(
    runtime_settings: Settings,
) -> None:
    application = runtime_settings.model_copy(
        update={
            "app_env": "testing",
            "frontend_origin": "http://localhost:8080",
            "database_url": "sqlite:///./test.db",
            "model_warmup_enabled": (
                not runtime_settings.model_warmup_enabled
            ),
        },
    )

    validate_model_settings_compatibility(
        application,
        runtime_settings,
    )


def test_multiple_mismatches_are_reported(
    runtime_settings: Settings,
) -> None:
    application = runtime_settings.model_copy(
        update={
            "embedding_model": "different-embedding",
            "reranker_model": "different-reranker",
            "ollama_model": "different-ollama-model",
        },
    )

    with pytest.raises(ModelSettingsMismatchError) as exc_info:
        validate_model_settings_compatibility(
            application,
            runtime_settings,
        )

    message = str(exc_info.value)

    assert "embedding_model" in message
    assert "reranker_model" in message
    assert "ollama_model" in message


def test_error_does_not_expose_api_key_values(
    runtime_settings: Settings,
) -> None:
    runtime_key = "runtime-secret-test-only-123"
    application_key = "application-secret-test-only-456"

    runtime = runtime_settings.model_copy(
        update={
            "llm_provider": "openai",
            "openai_api_key": runtime_key,
        },
    )
    application = runtime.model_copy(
        update={"openai_api_key": application_key},
    )

    with pytest.raises(ModelSettingsMismatchError) as exc_info:
        validate_model_settings_compatibility(
            application,
            runtime,
        )

    message = str(exc_info.value)

    assert "openai_api_key" in message
    assert runtime_key not in message
    assert application_key not in message


def test_validation_does_not_mutate_settings(
    runtime_settings: Settings,
) -> None:
    application = runtime_settings.model_copy(
        update={
            "llm_provider": " OLLAMA ",
            "ollama_base_url": (
                runtime_settings.ollama_base_url + "/"
            ),
        },
    )

    runtime_before = runtime_settings.model_dump()
    application_before = application.model_dump()

    validate_model_settings_compatibility(
        application,
        runtime_settings,
    )

    assert runtime_settings.model_dump() == runtime_before
    assert application.model_dump() == application_before


@pytest.mark.parametrize("provider", ["ollama", "openai"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_matching_reranker_devices_are_compatible(
    runtime_settings: Settings,
    provider: str,
    device: str,
) -> None:
    runtime = runtime_settings.model_copy(
        update={
            "llm_provider": provider,
            "reranker_device": device,
        },
    )

    application = runtime.model_copy()

    validate_model_settings_compatibility(
        application,
        runtime,
    )


@pytest.mark.parametrize("provider", ["ollama", "openai"])
@pytest.mark.parametrize(
    ("runtime_device", "application_device"),
    [
        ("cpu", "cuda:0"),
        ("cuda:0", "cpu"),
    ],
)
def test_reranker_device_mismatch_is_rejected(
    runtime_settings: Settings,
    provider: str,
    runtime_device: str,
    application_device: str,
) -> None:
    runtime = runtime_settings.model_copy(
        update={
            "llm_provider": provider,
            "reranker_device": runtime_device,
        },
    )

    application = runtime.model_copy(
        update={"reranker_device": application_device},
    )

    with pytest.raises(
        ModelSettingsMismatchError,
        match="reranker_device",
    ):
        validate_model_settings_compatibility(
            application,
            runtime,
        )
