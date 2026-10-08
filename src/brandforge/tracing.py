"""Langfuse tracing (BF-25): one trace per run, one span per node, one generation per model call.

How the pieces fit (the reasoning is in ADR 0019):

- `run_trace` opens the root span for a run. Its trace ID is made from the run's own `run_id`
  (`trace_id_for`), so it is known before the run starts, is stored in state and the result, and
  is the same on every machine for the same `run_id`.
- `node_span` opens a span for one graph node. The graph wraps every node in it
  (`brandforge.graph`).
- `generation` opens a Langfuse generation for one model call. The gateway core does this, so
  every provider is traced the same way and no adapter knows about tracing (`brandforge.llm`).

Spans nest through OpenTelemetry's context, which Langfuse's SDK is built on, so a generation
lands under the node that made it without anything being passed along. A model call made outside
a run (the baseline generator, a script) is not traced: there is no run trace to put it in.

Tracing must never change a run. It is off unless `settings.tracing_enabled` is true and both
Langfuse keys are set, and when it is on, an error inside Langfuse is logged as a warning and the
run carries on. Every function here is a no-op in the off state.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langfuse import Langfuse, propagate_attributes

from brandforge.config import Settings, get_settings
from brandforge.models import RunState

__all__ = ["Span", "generation", "node_span", "run_trace", "trace_id_for"]

logger = logging.getLogger(__name__)

RUN_SPAN_NAME = "brandforge.run"

_clients: dict[tuple[str, str, str], Langfuse] = {}
_clients_lock = threading.Lock()

# The client of the run being traced, set by `run_trace`. `None` outside a traced run, which is
# what turns `node_span` and `generation` into no-ops for the baseline and for tracing-off runs.
_active_client: ContextVar[Langfuse | None] = ContextVar("brandforge_trace_client", default=None)


def _make_client(settings: Settings) -> Langfuse:
    """Build the Langfuse client from settings. A named seam so tests can swap in a client that
    exports to memory instead of the network."""
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )


def _client_for(settings: Settings) -> Langfuse | None:
    """The shared client for these settings, or `None` when tracing is off or cannot start.

    One client is kept per key pair and host, because each one starts a background exporter.
    """
    if not (settings.tracing_enabled and settings.langfuse_configured):
        return None
    key = (
        settings.langfuse_public_key.get_secret_value(),
        settings.langfuse_secret_key.get_secret_value(),
        settings.langfuse_host,
    )
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            try:
                client = _make_client(settings)
            except Exception as exc:
                logger.warning("Langfuse tracing is off: could not start the client: %s", exc)
                return None
            _clients[key] = client
        return client


def trace_id_for(run_id: str, settings: Settings | None = None) -> str | None:
    """The Langfuse trace ID for a run, or `None` when tracing is off.

    It is derived from `run_id`, so the same run always maps to the same trace.
    """
    if _client_for(settings or get_settings()) is None:
        return None
    return Langfuse.create_trace_id(seed=run_id)


class Span:
    """A handle on an open Langfuse observation. Safe to call whether or not tracing is on.

    With tracing off, or if Langfuse failed to open the observation, it holds nothing and
    `update` does nothing. A failure inside Langfuse is logged, never raised into the run.
    """

    def __init__(self, observation: Any | None = None) -> None:
        self._observation = observation

    def update(self, **fields: Any) -> None:
        """Add or change fields on the observation (`output`, `level`, `usage_details`, ...)."""
        if self._observation is None:
            return
        try:
            self._observation.update(**fields)
        except Exception as exc:
            logger.warning("Could not update the Langfuse observation: %s", exc)


@contextmanager
def _observe(client: Langfuse, **kwargs: Any) -> Iterator[Span]:
    """Open an observation as the current one, close it when the block ends, never raise for it.

    An exception from the block is passed to Langfuse, which marks the observation as an error,
    and is then re-raised unchanged.
    """
    try:
        manager = client.start_as_current_observation(**kwargs)
        observation = manager.__enter__()
    except Exception as exc:
        logger.warning("Could not open a Langfuse observation: %s", exc)
        yield Span()
        return
    try:
        yield Span(observation)
    except BaseException as exc:
        _close(manager, exc)
        raise
    else:
        _close(manager, None)


def _close(manager: Any, exc: BaseException | None) -> None:
    try:
        if exc is None:
            manager.__exit__(None, None, None)
        else:
            manager.__exit__(type(exc), exc, exc.__traceback__)
    except Exception as close_exc:
        logger.warning("Could not close a Langfuse observation: %s", close_exc)


@contextmanager
def run_trace(state: RunState, settings: Settings | None = None) -> Iterator[Span]:
    """Trace one run: open the root span for `state["trace_id"]`, flush when the run ends.

    Does nothing when the state has no trace ID (tracing was off when the run started) or
    tracing is off now. Everything opened inside the block (nodes, model calls) becomes a child
    of the root span. Flushing at the end is what gets a short-lived CLI run's trace out before
    the process exits.
    """
    client = _client_for(settings or get_settings())
    trace_id = state["trace_id"]
    if client is None or trace_id is None:
        yield Span()
        return

    brand = state["brand"]
    brief = state["brief"]
    token = _active_client.set(client)
    try:
        with (
            _observe(
                client,
                name=RUN_SPAN_NAME,
                trace_context={"trace_id": trace_id},
                input={"brief": brief.model_dump(mode="json"), "brand": brand.id},
            ) as root,
            _propagated(
                trace_name=RUN_SPAN_NAME,
                tags=[f"brand:{brand.id}"],
                metadata={
                    "run_id": state["run_id"],
                    "brand_version": brand.version,
                    "rubric_version": brand.rubric.version,
                },
            ),
        ):
            yield root
    finally:
        _active_client.reset(token)
        _flush(client)


@contextmanager
def _propagated(**attributes: Any) -> Iterator[None]:
    """Attach these attributes to the current trace and every observation opened inside."""
    try:
        manager = propagate_attributes(**attributes)
        manager.__enter__()
    except Exception as exc:
        logger.warning("Could not set the Langfuse trace attributes: %s", exc)
        yield
        return
    try:
        yield
    except BaseException as exc:
        _close(manager, exc)
        raise
    else:
        _close(manager, None)


def _flush(client: Langfuse) -> None:
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Could not flush Langfuse traces: %s", exc)


@contextmanager
def node_span(name: str, state: RunState, *, version: str | None = None) -> Iterator[Span]:
    """A span for one graph node, under the run's root span. A no-op outside a traced run.

    `version` is the node's prompt version (for example `planner_v1`), so a trace says exactly
    which prompt file produced it.
    """
    client = _active_client.get()
    if client is None:
        yield Span()
        return
    with _observe(
        client,
        name=name,
        version=version,
        input={
            "variants": len(state["variants"]),
            "critiques": len(state["critiques"]),
            "revision_count": state["revision_count"],
            "has_plan": state["plan"] is not None,
        },
        metadata={"node": name, "revision_count": state["revision_count"]},
    ) as span:
        yield span


@contextmanager
def generation(
    *,
    name: str,
    model: str,
    provider: str,
    tier: str,
    prompt: str,
    system: str | None,
    max_output_tokens: int,
) -> Iterator[Span]:
    """A generation for one model call, under the node that made it. A no-op outside a traced
    run. The caller adds the reply and the usage with `Span.update` once it has them."""
    client = _active_client.get()
    if client is None:
        yield Span()
        return
    model_input: Any = prompt
    if system is not None:
        model_input = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    with _observe(
        client,
        name=name,
        as_type="generation",
        model=model,
        input=model_input,
        model_parameters={"max_output_tokens": max_output_tokens},
        metadata={"provider": provider, "tier": tier},
    ) as span:
        yield span
