"""The graph: planner then writer, with typed state and reducers."""

from typing import Any

import pytest

from brandforge.brands import load_brand
from brandforge.graph import PLANNER_NODE, WRITER_NODE, build_graph, run_graph
from brandforge.llm.base import GatewayError
from brandforge.models import Brief, Plan, RunError, RunState, Usage, Variant, new_run_state


@pytest.fixture
def brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )


class FakePlan:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        if self.error:
            raise self.error
        plan = Plan(
            audience="commuters",
            angle="save time",
            channels=list(state["brief"].channels),
            variants_per_channel=2,
        )
        return {"plan": plan, "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.002)}


class FakeWrite:
    def __init__(
        self, error: Exception | None = None, errors: list[RunError] | None = None
    ) -> None:
        self.error = error
        self.errors = errors or []
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        if self.error:
            raise self.error
        variant = Variant(id="search-1", channel="search", headline="H", body="B", cta="C")
        update: dict[str, Any] = {
            "variants": [variant],
            "usage": Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        }
        if self.errors:
            update["errors"] = self.errors
        return update


def test_graph_runs_the_planner_then_the_writer() -> None:
    graph = build_graph(FakePlan(), FakeWrite())

    nodes = graph.get_graph()
    assert set(nodes.nodes) == {"__start__", PLANNER_NODE, WRITER_NODE, "__end__"}
    edges = {(e.source, e.target) for e in nodes.edges}
    assert edges == {
        ("__start__", PLANNER_NODE),
        (PLANNER_NODE, WRITER_NODE),
        (WRITER_NODE, "__end__"),
    }


def test_run_graph_returns_variants_and_complete_status(brief: Brief) -> None:
    write = FakeWrite()

    state = run_graph(brief, load_brand("voltride"), plan=FakePlan(), write=write, run_id="run-1")

    assert state["run_id"] == "run-1"
    assert state["status"] == "complete"
    assert [v.id for v in state["variants"]] == ["search-1"]
    assert state["errors"] == []
    assert len(write.calls) == 1


def test_writer_receives_the_plan_the_planner_wrote(brief: Brief) -> None:
    write = FakeWrite()

    state = run_graph(brief, load_brand("voltride"), plan=FakePlan(), write=write)

    assert state["plan"] is not None
    assert state["plan"].angle == "save time"
    assert write.calls[0]["plan"] == state["plan"]


def test_writer_errors_are_kept_and_mark_the_run_partial(brief: Brief) -> None:
    error = RunError(node="writer", message="channel 'search': asked for 2 variants, got 1")

    state = run_graph(
        brief, load_brand("voltride"), plan=FakePlan(), write=FakeWrite(errors=[error])
    )

    assert state["status"] == "partial"
    assert state["errors"] == [error]
    assert len(state["variants"]) == 1


def test_usage_from_planner_and_writer_is_summed(brief: Brief) -> None:
    state = run_graph(brief, load_brand("voltride"), plan=FakePlan(), write=FakeWrite())

    assert state["usage"].input_tokens == 110
    assert state["usage"].output_tokens == 55
    assert state["usage"].cost_usd == pytest.approx(0.003)


def test_node_usage_is_added_to_existing_usage_by_the_reducer(brief: Brief) -> None:
    brand = load_brand("voltride")
    initial = new_run_state(brief, brand)
    initial["usage"] = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5)

    state = build_graph(FakePlan(), FakeWrite()).invoke(initial)

    assert state["usage"].input_tokens == 111
    assert state["usage"].output_tokens == 57
    assert state["usage"].cost_usd == pytest.approx(0.503)


def test_writer_errors_propagate_until_error_edges_exist(brief: Brief) -> None:
    # BF-22 turns this into a recorded RunError and a partial/failed status.
    with pytest.raises(GatewayError, match="boom"):
        run_graph(
            brief,
            load_brand("voltride"),
            plan=FakePlan(),
            write=FakeWrite(GatewayError("boom")),
        )


def test_planner_errors_stop_the_run_before_the_writer(brief: Brief) -> None:
    write = FakeWrite()

    with pytest.raises(GatewayError, match="no plan"):
        run_graph(
            brief,
            load_brand("voltride"),
            plan=FakePlan(GatewayError("no plan")),
            write=write,
        )

    assert write.calls == []
