"""The graph: planner then a baseline-wrapping node, with typed state and reducers."""

from typing import Any

import pytest

from brandforge.brands import load_brand
from brandforge.graph import BASELINE_NODE, PLANNER_NODE, build_graph, run_graph
from brandforge.llm.base import GatewayError
from brandforge.models import BrandProfile, Brief, Plan, RunState, Usage, Variant, new_run_state


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


class FakeGenerate:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[Brief, BrandProfile]] = []

    def __call__(self, brief: Brief, brand: BrandProfile) -> tuple[list[Variant], Usage]:
        self.calls.append((brief, brand))
        if self.error:
            raise self.error
        variant = Variant(id="baseline-1", channel="search", headline="H", body="B", cta="C")
        return [variant], Usage(input_tokens=100, output_tokens=50, cost_usd=0.001)


def test_graph_runs_the_planner_then_the_baseline_node() -> None:
    graph = build_graph(FakeGenerate(), FakePlan())

    nodes = graph.get_graph()
    assert set(nodes.nodes) == {"__start__", PLANNER_NODE, BASELINE_NODE, "__end__"}
    edges = {(e.source, e.target) for e in nodes.edges}
    assert edges == {
        ("__start__", PLANNER_NODE),
        (PLANNER_NODE, BASELINE_NODE),
        (BASELINE_NODE, "__end__"),
    }


def test_run_graph_returns_variants_usage_and_complete_status(brief: Brief) -> None:
    fake = FakeGenerate()
    brand = load_brand("voltride")

    state = run_graph(brief, brand, generate=fake, plan=FakePlan(), run_id="run-1")

    assert state["run_id"] == "run-1"
    assert state["status"] == "complete"
    assert [v.id for v in state["variants"]] == ["baseline-1"]
    assert state["errors"] == []
    assert fake.calls == [(brief, brand)]


def test_planner_writes_the_plan_into_state(brief: Brief) -> None:
    planner = FakePlan()

    state = run_graph(brief, load_brand("voltride"), generate=FakeGenerate(), plan=planner)

    assert state["plan"] is not None
    assert state["plan"].angle == "save time"
    assert state["plan"].channels == ["search"]
    assert planner.calls[0]["brief"] is brief


def test_usage_from_planner_and_baseline_is_summed(brief: Brief) -> None:
    state = run_graph(brief, load_brand("voltride"), generate=FakeGenerate(), plan=FakePlan())

    assert state["usage"].input_tokens == 110
    assert state["usage"].output_tokens == 55
    assert state["usage"].cost_usd == pytest.approx(0.003)


def test_node_usage_is_added_to_existing_usage_by_the_reducer(brief: Brief) -> None:
    brand = load_brand("voltride")
    initial = new_run_state(brief, brand)
    initial["usage"] = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5)

    state = build_graph(FakeGenerate(), FakePlan()).invoke(initial)

    assert state["usage"].input_tokens == 111
    assert state["usage"].output_tokens == 57
    assert state["usage"].cost_usd == pytest.approx(0.503)


def test_generator_errors_propagate_until_error_edges_exist(brief: Brief) -> None:
    # BF-22 turns this into a recorded RunError and a partial/failed status.
    with pytest.raises(GatewayError, match="boom"):
        run_graph(
            brief,
            load_brand("voltride"),
            generate=FakeGenerate(GatewayError("boom")),
            plan=FakePlan(),
        )


def test_planner_errors_stop_the_run_before_the_baseline(brief: Brief) -> None:
    generate = FakeGenerate()

    with pytest.raises(GatewayError, match="no plan"):
        run_graph(
            brief,
            load_brand("voltride"),
            generate=generate,
            plan=FakePlan(GatewayError("no plan")),
        )

    assert generate.calls == []
