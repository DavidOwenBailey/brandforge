"""The graph shell: one node wrapping the baseline, with typed state and reducers."""

import pytest

from brandforge.brands import load_brand
from brandforge.graph import BASELINE_NODE, build_graph, run_graph
from brandforge.llm.base import GatewayError
from brandforge.models import BrandProfile, Brief, Usage, Variant, new_run_state


@pytest.fixture
def brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )


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


def test_graph_has_the_single_baseline_node() -> None:
    graph = build_graph(FakeGenerate())

    assert set(graph.get_graph().nodes) == {"__start__", BASELINE_NODE, "__end__"}


def test_run_graph_returns_variants_usage_and_complete_status(brief: Brief) -> None:
    fake = FakeGenerate()
    brand = load_brand("voltride")

    state = run_graph(brief, brand, generate=fake, run_id="run-1")

    assert state["run_id"] == "run-1"
    assert state["status"] == "complete"
    assert [v.id for v in state["variants"]] == ["baseline-1"]
    assert state["usage"] == Usage(input_tokens=100, output_tokens=50, cost_usd=0.001)
    assert state["errors"] == []
    assert fake.calls == [(brief, brand)]


def test_node_usage_is_added_to_existing_usage_by_the_reducer(brief: Brief) -> None:
    brand = load_brand("voltride")
    initial = new_run_state(brief, brand)
    initial["usage"] = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5)

    state = build_graph(FakeGenerate()).invoke(initial)

    assert state["usage"].input_tokens == 101
    assert state["usage"].output_tokens == 52
    assert state["usage"].cost_usd == pytest.approx(0.501)


def test_generator_errors_propagate_until_error_edges_exist(brief: Brief) -> None:
    # BF-22 turns this into a recorded RunError and a partial/failed status.
    with pytest.raises(GatewayError, match="boom"):
        run_graph(brief, load_brand("voltride"), generate=FakeGenerate(GatewayError("boom")))
