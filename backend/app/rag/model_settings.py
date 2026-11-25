"""Validate application settings against shared model configuration."""

from backend.app.core.config import Settings


class ModelSettingsMismatchError(ValueError):
    """Application and process-wide model settings are inconsistent."""


def validate_model_settings_compatibility(
    application_settings: Settings,
    runtime_settings: Settings,
) -> None:
    """Reject model overrides unsupported by process-wide factories."""
    mismatches: list[str] = []

    common_fields = (
        "embedding_model",
        "sparse_model",
        "reranker_model",
        "llm_temperature",
    )

    for field in common_fields:
        if getattr(application_settings, field) != getattr(
            runtime_settings,
            field,
        ):
            mismatches.append(field)

    application_provider = (
        application_settings.llm_provider.strip().lower()
    )
    runtime_provider = (
        runtime_settings.llm_provider.strip().lower()
    )

    if application_provider != runtime_provider:
        mismatches.append("llm_provider")

    elif runtime_provider == "ollama":
        if (
            application_settings.ollama_base_url.rstrip("/")
            != runtime_settings.ollama_base_url.rstrip("/")
        ):
            mismatches.append("ollama_base_url")

        for field in ("ollama_model", "ollama_keep_alive"):
            if getattr(application_settings, field) != getattr(
                runtime_settings,
                field,
            ):
                mismatches.append(field)

    elif runtime_provider == "openai":
        for field in ("llm_model", "openai_api_key"):
            if getattr(application_settings, field) != getattr(
                runtime_settings,
                field,
            ):
                mismatches.append(field)

    if mismatches:
        fields = ", ".join(mismatches)

        raise ModelSettingsMismatchError(
            "Application model settings differ from process-wide "
            f"runtime settings: {fields}. "
            "Configure shared models through environment settings "
            "before starting the application."
        )
