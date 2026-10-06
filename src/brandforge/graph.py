"""The LangGraph pipeline: nodes, edges and the entry point the CLI and evals call.

For now the graph is a shell with one node that wraps the single-prompt baseline, so the
walking skeleton keeps working end to end. Later tasks replace that node with the planner,
writer, critic, router and reviser (BF-13 to BF-18).
"""

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from brandforge.baseline import generate_baseline
from brandforge.models import BrandProfile, Brief, RunState, Usage, Variant, new_run_state

# Same shape as `generate_baseline`; tests pass a fake.
Generator = Callable[[Brief, BrandProfile], tuple[list[Variant], Usage]]

BASELINE_NODE = "baseline"


def build_graph(
    generate: Generator | None = None,
) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    """Compile the graph. `generate` defaults to the baseline generator, looked up at call time."""
    generator = generate or generate_baseline

    def baseline_node(state: RunState) -> dict[str, Any]:
        variants, usage = generator(state["brief"], state["brand"])
        return {"variants": variants, "usage": usage, "status": "complete"}

    builder = StateGraph(RunState)
    builder.add_node(BASELINE_NODE, baseline_node)
    builder.add_edge(START, BASELINE_NODE)
    builder.add_edge(BASELINE_NODE, END)
    return builder.compile()


def run_graph(
    brief: Brief,
    brand: BrandProfile,
    *,
    generate: Generator | None = None,
    run_id: str | None = None,
) -> RunState:
    """Run one brief through the graph and return the final state."""
    graph = build_graph(generate)
    return cast(RunState, graph.invoke(new_run_state(brief, brand, run_id=run_id)))
