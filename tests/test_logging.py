"""Structured logging (BF-26). Logs are captured on a stream, never the real stderr."""

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict
from typer.testing import CliRunner

from brandforge.brands import load_brand
from brandforge.config import Settings
from brandforge.graph import run_graph
from brandforge.interfaces import cli
from brandforge.logging import bind_run, configure_logging, get_logger
from brandforge.models import Brief, Critique, Plan, RunState, Usage, Variant

runner = CliRunner()


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def _brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    """JSON logs for one test, torn down so later tests are not stuck with our handler."""
    stream = io.StringIO()
    root = logging.getLogger()
    package = logging.getLogger("brandforge")
    previous_root = root.level
    previous_package = package.level
    configure_logging(IsolatedSettings(), stream=stream)
    yield stream
    for handler in list(root.handlers):
        if handler.name == "brandforge":
            root.removeHandler(handler)
            handler.close()
    root.setLevel(previous_root)
    package.setLevel(previous_package)


def _events(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_a_log_line_is_json_with_the_standard_fields(log_stream: io.StringIO) -> None:
    get_logger("brandforge.demo").info("hello")

    (event,) = _events(log_stream)
    assert event["event"] == "hello"
    assert event["level"] == "info"
    assert event["logger"] == "brandforge.demo"
    assert "T" in event["timestamp"]
    assert event["timestamp"].endswith("Z")


def test_debug_is_dropped_at_info(log_stream: io.StringIO) -> None:
    get_logger("brandforge.demo").debug("hidden")
    get_logger("brandforge.demo").info("shown")

    assert [event["event"] for event in _events(log_stream)] == ["shown"]


def test_the_level_and_format_come_from_settings() -> None:
    stream = io.StringIO()
    root = logging.getLogger()
    package = logging.getLogger("brandforge")
    previous = package.level
    try:
        configure_logging(IsolatedSettings(log_level="DEBUG", log_format="console"), stream=stream)
        get_logger("brandforge.demo").debug("visible")
        text = stream.getvalue()
    finally:
        for handler in list(root.handlers):
            if handler.name == "brandforge":
                root.removeHandler(handler)
                handler.close()
        package.setLevel(previous)

    assert "visible" in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text.strip().splitlines()[-1])


def test_bind_run_adds_the_run_id_to_structlog_and_stdlib(log_stream: io.StringIO) -> None:
    with bind_run("run-1", trace_id="trace-1"):
        get_logger("brandforge.demo").info("from-structlog")
        logging.getLogger("brandforge.demo").warning("from-stdlib")

    struct, std = _events(log_stream)
    assert struct["event"] == "from-structlog"
    assert struct["run_id"] == "run-1"
    assert struct["trace_id"] == "trace-1"
    assert std["event"] == "from-stdlib"
    assert std["run_id"] == "run-1"
    assert std["trace_id"] == "trace-1"
    assert std["level"] == "warning"


def test_an_untraced_run_does_not_bind_a_null_trace_id(log_stream: io.StringIO) -> None:
    with bind_run("run-1"):
        get_logger("brandforge.demo").info("inside")

    (event,) = _events(log_stream)
    assert event["run_id"] == "run-1"
    assert "trace_id" not in event


def test_bind_run_resets_after_the_block_and_after_an_exception(log_stream: io.StringIO) -> None:
    with bind_run("outer", trace_id="trace-outer"):
        with bind_run("inner", trace_id="trace-inner"):
            get_logger("brandforge.demo").info("inner")
        get_logger("brandforge.demo").info("outer")
    try:
        with bind_run("exploded"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    get_logger("brandforge.demo").info("after")

    inner, outer, after = _events(log_stream)
    assert inner["run_id"] == "inner"
    assert inner["trace_id"] == "trace-inner"
    assert outer["run_id"] == "outer"
    assert outer["trace_id"] == "trace-outer"
    assert "run_id" not in after
    assert "trace_id" not in after


def test_a_secret_field_is_redacted_and_the_message_is_not(log_stream: io.StringIO) -> None:
    get_logger("brandforge.demo").warning(
        "the brief says secret", api_key="sk-real", nested={"secret": "still-here"}
    )

    (event,) = _events(log_stream)
    assert event["event"] == "the brief says secret"
    assert event["api_key"] == "***"
    assert event["nested"] == {"secret": "still-here"}
    assert "sk-real" not in log_stream.getvalue()


def _plan(state: RunState) -> dict[str, Any]:
    return {
        "plan": Plan(
            audience="commuters",
            angle="save time",
            channels=list(state["brief"].channels),
            variants_per_channel=1,
        ),
        "usage": Usage(),
    }


def _write(message: str) -> Any:
    def write(_state: RunState) -> dict[str, Any]:
        logging.getLogger("brandforge.demo").info(message)
        variant = Variant(id="search-1", channel="search", headline="H", body="B", cta="C")
        return {"variants": [variant], "usage": Usage()}

    return write


def _critique(_state: RunState) -> dict[str, Any]:
    return {
        "critiques": [
            Critique(variant_id="search-1", scores={"voice": 5}, overall=5.0, passed=True)
        ],
        "usage": Usage(),
    }


def test_a_run_binds_its_id_on_every_line_and_the_next_run_gets_its_own(
    log_stream: io.StringIO,
) -> None:
    brand = load_brand("voltride")
    run_graph(
        _brief(), brand, plan=_plan, write=_write("planning-a"), critique=_critique, run_id="run-a"
    )
    run_graph(
        _brief(), brand, plan=_plan, write=_write("planning-b"), critique=_critique, run_id="run-b"
    )
    get_logger("brandforge.demo").info("between-runs")

    events = _events(log_stream)
    by_run: dict[str, set[str]] = {}
    for event in events:
        if "run_id" in event:
            by_run.setdefault(event["run_id"], set()).add(event["event"])

    assert by_run["run-a"] >= {"run_started", "planning-a", "run_finished"}
    assert by_run["run-b"] >= {"run_started", "planning-b", "run_finished"}
    assert "planning-b" not in by_run["run-a"]
    started = next(event for event in events if event["event"] == "run_started")
    assert started["brand_id"] == "voltride"
    finished = next(
        event for event in events if event["event"] == "run_finished" and event["run_id"] == "run-a"
    )
    assert finished["status"] == "complete"
    assert finished["tokens"] == 0
    assert finished["cost_usd"] == 0
    assert "trace_id" not in started  # tracing is off in tests
    assert events[-1]["event"] == "between-runs"
    assert "run_id" not in events[-1]


def test_a_failed_node_still_finishes_the_run_log(log_stream: io.StringIO) -> None:
    def boom(_state: RunState) -> dict[str, Any]:
        raise RuntimeError("planner blew up")

    state = run_graph(_brief(), load_brand("voltride"), plan=boom, run_id="run-fail")

    events = _events(log_stream)
    warning = next(event for event in events if "planner blew up" in event["event"])
    assert warning["run_id"] == "run-fail"
    assert warning["level"] == "warning"
    finished = next(event for event in events if event["event"] == "run_finished")
    assert finished["run_id"] == "run-fail"
    assert finished["status"] == "failed"
    assert state["result"] is not None
    assert state["result"].status == "failed"


def test_a_traced_run_logs_its_trace_id(
    log_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "brandforge.graph.tracing.trace_id_for", lambda _run_id, _settings: "trace-abc"
    )

    run_graph(
        _brief(),
        load_brand("voltride"),
        plan=_plan,
        write=_write("planning"),
        critique=_critique,
        run_id="run-traced",
    )

    started = next(event for event in _events(log_stream) if event["event"] == "run_started")
    assert started["run_id"] == "run-traced"
    assert started["trace_id"] == "trace-abc"
    seen: list[Settings] = []
    monkeypatch.setattr(cli, "configure_logging", lambda settings: seen.append(settings))

    result = runner.invoke(cli.app, ["generate"])

    assert result.exit_code != 0
    assert seen
