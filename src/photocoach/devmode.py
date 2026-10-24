"""Developer mode switch: exposes the local mock coach for development and tests.

Normal launches keep it off. It is enabled only by the ``--dev`` command-line
argument or the ``PHOTOCOACH_DEV=1`` environment variable.
"""
from __future__ import annotations

import os

ENVIRONMENT_VARIABLE = "PHOTOCOACH_DEV"


def dev_mode_enabled(explicit: bool | None = None) -> bool:
    """An explicit True/False wins; ``None`` defers to the environment (default off)."""
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(ENVIRONMENT_VARIABLE, "") == "1"
