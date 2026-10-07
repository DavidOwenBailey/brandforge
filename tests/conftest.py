"""Shared test fixtures."""

import pytest

from brandforge.llm import gateway


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the gateway's retry waits instead of sleeping, so no test is ever slowed by them.

    Request it by name in a test to assert on the waits.
    """
    waits: list[float] = []
    monkeypatch.setattr(gateway, "_sleep", waits.append)
    return waits
