"""Shared configuration for integration tests.

Tests marked ``integration`` use real models (downloaded on first run) and a
Qdrant instance. They are skipped unless ``RUN_QDRANT_INTEGRATION=1`` is set.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: uses real embedding models and a Qdrant instance",
    )
