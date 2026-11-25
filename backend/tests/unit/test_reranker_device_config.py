"""Unit tests for reranker device configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


def test_reranker_device_defaults_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RERANKER_DEVICE", raising=False)

    settings = Settings(_env_file=None)

    assert settings.reranker_device == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_explicit_reranker_device_is_accepted(
    device: str,
) -> None:
    settings = Settings(
        _env_file=None,
        reranker_device=device,
    )

    assert settings.reranker_device == device


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_reranker_device_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setenv("RERANKER_DEVICE", device)

    settings = Settings(_env_file=None)

    assert settings.reranker_device == device


@pytest.mark.parametrize(
    "device",
    ["", "auto", "cuda", "cuda:1", "mps", "gpu", "CPU", " cuda:0 "],
)
def test_invalid_reranker_device_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setenv("RERANKER_DEVICE", device)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert any(
        error["loc"] == ("reranker_device",)
        for error in exc_info.value.errors()
    )


def test_explicit_device_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_DEVICE", "cuda:0")

    settings = Settings(
        _env_file=None,
        reranker_device="cpu",
    )

    assert settings.reranker_device == "cpu"


def test_reranker_device_from_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RERANKER_DEVICE", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "RERANKER_DEVICE=cuda:0\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.reranker_device == "cuda:0"


def test_device_environment_overrides_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RERANKER_DEVICE=cuda:0\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("RERANKER_DEVICE", "cpu")

    settings = Settings(_env_file=env_file)

    assert settings.reranker_device == "cpu"
