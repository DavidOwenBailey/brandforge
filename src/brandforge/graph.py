"""The LangGraph pipeline: nodes, edges and the entry point the CLI and evals call.

The graph runs the planner (BF-13) and then the writer (BF-14). The critic, router and reviser
follow (BF-15 to BF-17) and the assembler (BF-18) will own the final status; until then the
writer's node marks the run `complete`, or `partial` if the writer recorded errors. The
single-prompt baseline is no longer part of the graph: the evals call it directly.
"""

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from brandforge.agents.planner import plan_brief
from brandforge.agents.writer import write_variants
from brandforge.models import BrandProfile, Brief, RunState, new_run_state

# Same shape as `plan_brief` and `write_variants`: read state, return a partial state update.
Node = Callable[[RunState], dict[str, Any]]

PLANNER_NODE = "planner"
WRITER_NODE = "writer"


def build_graph(
    plan: Node | None = None,
    write: Node | None = None,
) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    """Compile the graph. `plan` and `write` default to the real agents, looked up at call
    time so tests can replace them."""
    planner = plan or plan_brief
    writer = write or write_variants

    def planner_node(state: RunState) -> dict[str, Any]:
        return planner(state)

    def writer_node(state: RunState) -> dict[str, Any]:
        update = writer(state)
        return {**update, "status": "partial" if update.get("errors") else "complete"}

    builder = StateGraph(RunState)
    builder.add_node(PLANNER_NODE, planner_node)
    builder.add_node(WRITER_NODE, writer_node)
    builder.add_edge(START, PLANNER_NODE)
    builder.add_edge(PLANNER_NODE, WRITER_NODE)
    builder.add_edge(WRITER_NODE, END)
    return builder.compile()


def run_graph(
    brief: Brief,
    brand: BrandProfile,
    *,
    plan: Node | None = None,
    write: Node | None = None,
    run_id: str | None = None,
) -> RunState:
    """Run one brief through the graph and return the final state."""
    graph = build_graph(plan, write)
    return cast(RunState, graph.invoke(new_run_state(brief, brand, run_id=run_id)))
