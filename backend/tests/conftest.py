"""Shared test configuration for the backend test suite."""

import os

# Disable automatic startup model warm-up before test modules
# import the application and initialize cached settings.
#
# Dedicated lifecycle tests can explicitly enable warm-up
# and inject controlled model dependencies.
os.environ["MODEL_WARMUP_ENABLED"] = "false"
