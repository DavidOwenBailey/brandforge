"""Shared test fixtures."""

from collections.abc import Iterator
from typing import Any

import pytest

from brandforge.config import get_settings
from brandforge.llm import gateway
from brandforge.models import RunState


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the gateway's retry waits instead of sleeping, so no test is ever slowed by them.

    Request it by name in a test to assert on the waits.
    """
    waits: list[float] = []
    monkeypatch.setattr(gateway, "_sleep", waits.append)
    return waits


@pytest.fixture(autouse=True)
def no_real_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph runs must not open Chroma or download the embedding model.

    The retriever node still runs. It returns no examples, which is the empty result the
    writer already accepts. A test that wants a different retriever passes `retrieve=`, or
    replaces `brandforge.graph.retrieve_examples` itself.
    """

    def empty(_state: RunState) -> dict[str, Any]:
        return {"examples": []}

    monkeypatch.setattr("brandforge.graph.retrieve_examples", empty)


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every test off the network: Langfuse tracing is off unless a test turns it on.

    A developer's .env may hold real Langfuse keys, and the CLI and graph tests read settings
    from it. An environment variable beats .env, and a test that wants tracing passes
    `tracing_enabled=True` to its own `Settings`, which beats the environment.
    """
    monkeypatch.setenv("BRANDFORGE_TRACING_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
