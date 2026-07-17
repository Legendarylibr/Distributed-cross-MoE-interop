"""Pytest defaults: lab security profile so unit/e2e stacks remain usable.

Hardened ``secure`` posture is covered in ``tests/test_security.py``.
"""

from __future__ import annotations

import os

import pytest

# Ensure subprocesses (e2e) inherit lab unless a test overrides.
os.environ.setdefault("CEI_SECURITY_PROFILE", "lab")


@pytest.fixture(autouse=True)
def _cei_lab_security(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEI_SECURITY_PROFILE", "lab")
    from cei import security

    security.reset_config_cache()
    yield
    security.reset_config_cache()
