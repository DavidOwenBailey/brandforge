"""The LangGraph pipeline: nodes, edges and the entry point the CLI and evals call.

The graph runs the planner (BF-13) and then a node that still wraps the single-prompt
baseline, so the walking skeleton keeps working end to end. The baseline ignores the plan
until the writer replaces it (BF-14); the critic, router and reviser follow (BF-15 to BF-17).
"""

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from brandforge.agents.planner import plan_brief
from brandforge.baseline import generate_baseline
from brandforge.models import BrandProfile, Brief, RunState, Usage, Variant, new_run_state

# Same shape as `generate_baseline`; tests pass a fake.
Generator = Callable[[Brief, BrandProfile], tuple[list[Variant], Usage]]
# Same shape as `plan_brief`: reads state, returns a partial state update.
Planner = Callable[[RunState], dict[str, Any]]

PLANNER_NODE = "planner"
BASELINE_NODE = "baseline"


def build_graph(
    generate: Generator | None = None,
    plan: Planner | None = None,
) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    """Compile the graph. `generate` and `plan` default to the real baseline generator and
    planner, looked up at call time so tests can replace them."""
    generator = generate or generate_baseline
    planner = plan or plan_brief

    def planner_node(state: RunState) -> dict[str, Any]:
        return planner(state)

    def baseline_node(state: RunState) -> dict[str, Any]:
        variants, usage = generator(state["brief"], state["brand"])
        return {"variants": variants, "usage": usage, "status": "complete"}

    builder = StateGraph(RunState)
    builder.add_node(PLANNER_NODE, planner_node)
    builder.add_node(BASELINE_NODE, baseline_node)
    builder.add_edge(START, PLANNER_NODE)
    builder.add_edge(PLANNER_NODE, BASELINE_NODE)
    builder.add_edge(BASELINE_NODE, END)
    return builder.compile()


def run_graph(
    brief: Brief,
    brand: BrandProfile,
    *,
    generate: Generator | None = None,
    plan: Planner | None = None,
    run_id: str | None = None,
) -> RunState:
    """Run one brief through the graph and return the final state."""
    graph = build_graph(generate, plan)
    return cast(RunState, graph.invoke(new_run_state(brief, brand, run_id=run_id)))
