"""Shared configuration for integration tests.

Tests marked ``integration`` use real models (downloaded on first run) and a
Qdrant instance. Deselect them with ``pytest -m "not integration"``.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: uses real embedding models and a Qdrant instance",
    )
