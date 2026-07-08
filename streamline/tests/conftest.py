from __future__ import annotations

import sys

import pytest


def pytest_sessionstart(session):
    if sys.version_info < (3, 10):
        pytest.exit("STREAMLINE tests require Python 3.10 or newer.", returncode=2)
