"""Langfuse tracing (BF-25). The real Langfuse SDK runs, but its spans go to an in-memory
exporter, so nothing touches the network. Models are faked at the adapter boundary."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from brandforge import graph, tracing
from brandforge.brands import load_brand
from brandforge.checkpointing import open_checkpointer
from brandforge.config import Settings
from brandforge.graph import run_graph
from brandforge.llm.base import RawCompletion, StructuredOutputError
from brandforge.llm.gateway import complete_structured
from brandforge.models import (
    Brief,
    Critique,
    Plan,
    RunError,
    RunState,
    Usage,
    Variant,
    new_run_state,
)

VALID_VARIANT = (
    '{"id": "v1", "channel": "search", "headline": "Ride further", '
    '"body": "Built for the long way home.", "cta": "Shop now"}'
)


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


# Langfuse keeps one client per public key for the whole process, so each test uses a key of its
# own: a shared one would hand a later test the earlier test's client and exporter.
_public_key = "pk-test"


def _settings(**fields: Any) -> Settings:
    """Settings from field or alias names (the Langfuse keys are set by their env-var names)."""
    return IsolatedSettings.model_validate(fields)


def _on() -> Settings:
    """Settings with tracing switched on and both Langfuse keys set."""
    return _settings(
        tracing_enabled=True, LANGFUSE_PUBLIC_KEY=_public_key, LANGFUSE_SECRET_KEY="sk-test"
    )


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Tracing on, with spans collected in memory. Patches the graph's settings to match."""
    global _public_key
    _public_key = f"pk-test-{uuid4().hex}"
    memory = InMemorySpanExporter()
    monkeypatch.setattr(
        tracing,
        "_make_client",
        lambda settings: Langfuse(
            public_key=_public_key,
            secret_key="sk-test",
            host="http://localhost:1",
            span_exporter=memory,
        ),
    )
    monkeypatch.setattr(graph, "get_settings", _on)
    tracing._clients.clear()
    yield memory
    for client in tracing._clients.values():
        client.shutdown()
    tracing._clients.clear()


def _spans(exporter: InMemorySpanExporter) -> dict[str, list[ReadableSpan]]:
    by_name: dict[str, list[ReadableSpan]] = {}
    for span in exporter.get_finished_spans():
        by_name.setdefault(span.name, []).append(span)
    return by_name


def _attr(span: ReadableSpan, key: str) -> Any:
    return (span.attributes or {}).get(f"langfuse.observation.{key}")


def _trace_id(span: ReadableSpan) -> str:
    return f"{span.context.trace_id:032x}" if span.context else ""


def _parent_id(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent else None


# --- fake agents ---------------------------------------------------------------------


@pytest.fixture
def brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )


def _plan(state: RunState) -> dict[str, Any]:
    plan = Plan(
        audience="commuters",
        angle="save time",
        channels=list(state["brief"].channels),
        variants_per_channel=1,
    )
    return {"plan": plan, "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.002)}


def _write(state: RunState) -> dict[str, Any]:
    variant = Variant(id="search-1", channel="search", headline="H", body="B", cta="C")
    return {"variants": [variant], "usage": Usage()}


def _critique(state: RunState) -> dict[str, Any]:
    critiques = [
        Critique(variant_id=v.id, scores={"voice": 5}, overall=5.0, passed=True)
        for v in state["variants"]
    ]
    return {"critiques": critiques, "usage": Usage()}


def _fail(state: RunState) -> dict[str, Any]:
    raise RuntimeError("boom")


# --- tracing off ---------------------------------------------------------------------


def test_tracing_needs_the_switch_and_both_keys() -> None:
    assert not IsolatedSettings().langfuse_configured
    assert not _settings(LANGFUSE_PUBLIC_KEY="pk").langfuse_configured
    assert _settings(LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk").langfuse_configured

    keys_only = _settings(LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk")
    assert tracing.trace_id_for("run-1", keys_only) is None  # the env switch is off in tests
    assert tracing.trace_id_for("run-1", _settings(tracing_enabled=True)) is None


def test_a_run_without_tracing_has_no_trace_id(brief: Brief) -> None:
    state = run_graph(brief, load_brand("voltride"), plan=_plan, write=_write, critique=_critique)

    assert state["trace_id"] is None
    assert state["result"] is not None
    assert state["result"].trace_id is None
    assert state["status"] == "complete"


# --- tracing on: the run trace and node spans ----------------------------------------


def test_a_run_is_one_trace_with_a_span_per_node(
    exporter: InMemorySpanExporter, brief: Brief
) -> None:
    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=_plan,
        write=_write,
        critique=_critique,
        run_id="run-1",
    )

    expected = Langfuse.create_trace_id(seed="run-1")
    assert state["trace_id"] == expected
    assert state["result"] is not None
    assert state["result"].trace_id == expected

    spans = _spans(exporter)
    assert set(spans) == {"brandforge.run", "planner", "writer", "critic", "assembler"}
    assert {_trace_id(s) for group in spans.values() for s in group} == {expected}

    root = spans["brandforge.run"][0]
    for node in ("planner", "writer", "critic", "assembler"):
        (span,) = spans[node]
        assert _parent_id(span) == root.context.span_id


def test_a_checkpointed_run_is_traced_the_same_way(
    exporter: InMemorySpanExporter, brief: Brief, tmp_path: Path
) -> None:
    with open_checkpointer(tmp_path / "checkpoints.sqlite") as checkpointer:
        state = run_graph(
            brief,
            load_brand("voltride"),
            plan=_plan,
            write=_write,
            critique=_critique,
            checkpointer=checkpointer,
        )

    spans = _spans(exporter)
    assert {_trace_id(s) for group in spans.values() for s in group} == {state["trace_id"]}
    assert set(spans) == {"brandforge.run", "planner", "writer", "critic", "assembler"}


def test_the_trace_carries_the_brand_and_the_run_summary(
    exporter: InMemorySpanExporter, brief: Brief
) -> None:
    run_graph(
        brief,
        load_brand("voltride"),
        plan=_plan,
        write=_write,
        critique=_critique,
        run_id="run-1",
    )

    (root,) = _spans(exporter)["brandforge.run"]
    attributes = root.attributes or {}
    assert attributes["langfuse.trace.name"] == "brandforge.run"
    assert attributes["langfuse.trace.metadata.run_id"] == "run-1"
    assert "langfuse.trace.metadata.brand_version" in attributes
    assert attributes["langfuse.trace.tags"] == ("brand:voltride",)
    assert '"status": "complete"' in _attr(root, "output")


def test_a_node_span_records_the_prompt_version_and_a_summary(
    exporter: InMemorySpanExporter, brief: Brief
) -> None:
    run_graph(brief, load_brand("voltride"), plan=_plan, write=_write, critique=_critique)

    spans = _spans(exporter)
    (planner,) = spans["planner"]
    (assembler,) = spans["assembler"]
    assert (planner.attributes or {})["langfuse.version"] == "planner_v1"
    assert '"tokens": 15' in _attr(planner, "output")
    assert "langfuse.version" not in (assembler.attributes or {})  # no prompt, so no version


def test_a_failed_node_is_an_error_span_and_the_run_still_ends_at_the_assembler(
    exporter: InMemorySpanExporter, brief: Brief
) -> None:
    state = run_graph(brief, load_brand("voltride"), plan=_plan, write=_fail)

    spans = _spans(exporter)
    (writer,) = spans["writer"]
    assert _attr(writer, "level") == "ERROR"
    assert "boom" in _attr(writer, "status_message")
    assert _attr(spans["planner"][0], "level") is None
    assert "assembler" in spans  # the error edge still sends the run on
    assert state["status"] == "failed"


def test_every_revision_is_its_own_span(exporter: InMemorySpanExporter, brief: Brief) -> None:
    calls = {"critic": 0}

    def critique(state: RunState) -> dict[str, Any]:
        calls["critic"] += 1
        passed = calls["critic"] > 1
        return {
            "critiques": [
                Critique(
                    variant_id=v.id,
                    scores={"voice": 5 if passed else 2},
                    overall=5.0 if passed else 2.0,
                    passed=passed,
                    fixes=[] if passed else ["Tighten it"],
                )
                for v in state["variants"]
            ],
            "usage": Usage(),
        }

    def revise(state: RunState) -> dict[str, Any]:
        return {
            "variants": state["variants"],
            "critiques": [],
            "revision_count": state["revision_count"] + 1,
            "usage": Usage(),
        }

    run_graph(
        brief,
        load_brand("voltride"),
        plan=_plan,
        write=_write,
        critique=critique,
        revise=revise,
    )

    spans = _spans(exporter)
    assert len(spans["critic"]) == 2
    assert len(spans["reviser"]) == 1


# --- tracing on: model calls ---------------------------------------------------------


class ScriptedAdapter:
    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        schema: type[BaseModel],
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> RawCompletion:
        text = self._replies.pop(0)
        return RawCompletion(text=text, input_tokens=100, output_tokens=50, outcome="complete")


def _traced_state() -> RunState:
    state = new_run_state(
        Brief(product="p", audience="a", objective="conversion", channels=["search"]),
        load_brand("voltride"),
        run_id="run-1",
    )
    state["trace_id"] = tracing.trace_id_for("run-1", _on())
    return state


def test_a_model_call_is_a_generation_under_its_node(exporter: InMemorySpanExporter) -> None:
    state = _traced_state()

    with tracing.run_trace(state, _on()), tracing.node_span("writer", state):
        complete_structured(
            "write copy",
            Variant,
            "fast",
            adapter=ScriptedAdapter(VALID_VARIANT),
            settings=_on(),
        )

    spans = _spans(exporter)
    (generation,) = spans["Variant"]
    (writer,) = spans["writer"]
    assert _parent_id(generation) == writer.context.span_id
    assert _attr(generation, "type") == "generation"
    assert _attr(generation, "model.name") == "claude-haiku-4-5-20251001"
    assert _attr(generation, "input") == "write copy"
    assert _attr(generation, "output") == VALID_VARIANT
    assert '"input": 100' in _attr(generation, "usage_details")
    assert '"total": 150' in _attr(generation, "usage_details")
    assert '"total": 0.00035' in _attr(generation, "cost_details")
    assert _attr(generation, "metadata.tier") == "fast"
    assert _attr(generation, "metadata.provider") == "anthropic"
    assert _attr(generation, "metadata.outcome") == "complete"


def test_a_schema_repair_is_a_second_generation_and_the_first_is_an_error(
    exporter: InMemorySpanExporter,
) -> None:
    state = _traced_state()

    with tracing.run_trace(state, _on()):
        complete_structured(
            "write copy",
            Variant,
            "fast",
            adapter=ScriptedAdapter("not json", VALID_VARIANT),
            settings=_on(),
        )

    first, second = sorted(_spans(exporter)["Variant"], key=lambda s: s.start_time or 0)
    assert first.status.status_code.name == "ERROR"
    assert second.status.status_code.name == "UNSET"
    assert _attr(first, "output") == "not json"  # the failed reply is still on the trace


def test_a_reply_that_never_validates_ends_the_generation_as_an_error(
    exporter: InMemorySpanExporter,
) -> None:
    state = _traced_state()

    with (
        tracing.run_trace(state, _on()),
        pytest.raises(StructuredOutputError),
    ):
        complete_structured(
            "write copy",
            Variant,
            "strong",
            adapter=ScriptedAdapter("nope", "still nope"),
            settings=_on(),
        )

    generations = _spans(exporter)["Variant"]
    assert len(generations) == 2
    assert all(g.status.status_code.name == "ERROR" for g in generations)


def test_model_calls_outside_a_traced_run_are_not_traced(exporter: InMemorySpanExporter) -> None:
    complete_structured(
        "write copy",
        Variant,
        "fast",
        adapter=ScriptedAdapter(VALID_VARIANT),
        settings=_on(),
    )

    assert exporter.get_finished_spans() == ()


def test_nothing_is_traced_with_tracing_off(exporter: InMemorySpanExporter) -> None:
    state = _traced_state()
    off = IsolatedSettings()

    with tracing.run_trace(state, off), tracing.node_span("writer", state):
        complete_structured(
            "write copy", Variant, "fast", adapter=ScriptedAdapter(VALID_VARIANT), settings=off
        )

    assert exporter.get_finished_spans() == ()


# --- a broken Langfuse must not break a run ------------------------------------------


def test_a_client_that_cannot_start_turns_tracing_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(settings: Settings) -> Langfuse:
        raise RuntimeError("no network stack")

    monkeypatch.setattr(tracing, "_make_client", broken)
    tracing._clients.clear()

    with caplog.at_level("WARNING"):
        assert tracing.trace_id_for("run-1", _on()) is None

    assert "could not start the client" in caplog.text


def test_errors_inside_langfuse_do_not_change_the_run(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, brief: Brief
) -> None:
    client = tracing._client_for(_on())
    assert client is not None

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("langfuse is down")

    monkeypatch.setattr(client, "start_as_current_observation", explode)
    monkeypatch.setattr(client, "flush", explode)

    state = run_graph(brief, load_brand("voltride"), plan=_plan, write=_write, critique=_critique)

    assert state["status"] == "complete"
    assert state["errors"] == []


def test_a_node_error_still_reaches_the_caller_through_the_span(
    exporter: InMemorySpanExporter,
) -> None:
    state = _traced_state()

    with (
        tracing.run_trace(state, _on()),
        pytest.raises(ValueError, match="bad"),
        tracing.node_span("assembler", state),
    ):
        raise ValueError("bad")

    (assembler,) = _spans(exporter)["assembler"]
    assert assembler.status.status_code.name == "ERROR"


def test_a_span_summary_keeps_counts_and_totals_not_contents() -> None:
    # `_summarise` is what puts a node's result on its span, so check what it keeps and drops.
    update = {
        "variants": [object(), object()],
        "usage": Usage(input_tokens=3, output_tokens=4, cost_usd=0.5),
        "errors": [RunError(node="writer", message="x", fatal=True)],
        "revision_count": 1,
    }

    summary = graph._summarise(update)

    assert summary == {
        "variants": 2,
        "tokens": 7,
        "cost_usd": 0.5,
        "errors": 1,
        "revision_count": 1,
    }
