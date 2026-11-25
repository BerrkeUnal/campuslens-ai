from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings, get_settings


def test_model_warmup_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_WARMUP_ENABLED", raising=False)

    settings = Settings(_env_file=None)

    assert settings.model_warmup_enabled is True


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("TRUE", True),
        ("FALSE", False),
    ],
)
def test_model_warmup_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("MODEL_WARMUP_ENABLED", raw_value)

    settings = Settings(_env_file=None)

    assert settings.model_warmup_enabled is expected


@pytest.mark.parametrize(
    "raw_value",
    ["invalid", "", "maybe"],
)
def test_invalid_model_warmup_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    monkeypatch.setenv("MODEL_WARMUP_ENABLED", raw_value)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert any(
        error["loc"] == ("model_warmup_enabled",)
        for error in exc_info.value.errors()
    )


def test_explicit_setting_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_WARMUP_ENABLED", "true")

    settings = Settings(
        _env_file=None,
        model_warmup_enabled=False,
    )

    assert settings.model_warmup_enabled is False


def test_model_warmup_setting_from_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MODEL_WARMUP_ENABLED", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_WARMUP_ENABLED=false\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.model_warmup_enabled is False


def test_environment_overrides_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_WARMUP_ENABLED=true\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL_WARMUP_ENABLED", "false")

    settings = Settings(_env_file=env_file)

    assert settings.model_warmup_enabled is False


def test_cached_settings_preserve_value_until_cache_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_WARMUP_ENABLED", "false")

    get_settings.cache_clear()

    try:
        first = get_settings()

        assert first.model_warmup_enabled is False

        monkeypatch.setenv("MODEL_WARMUP_ENABLED", "true")

        second = get_settings()

        assert second is first
        assert second.model_warmup_enabled is False

        get_settings.cache_clear()

        third = get_settings()

        assert third is not first
        assert third.model_warmup_enabled is True

    finally:
        get_settings.cache_clear()
