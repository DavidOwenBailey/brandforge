"""The LangGraph pipeline: nodes, edges and the entry point the CLI and evals call.

The graph runs the planner (BF-13), the writer (BF-14) and the brand critic (BF-15). The router
and reviser follow (BF-16, BF-17) and the assembler (BF-18) will own the final status; until
then the writer's node marks the run `complete`, or `partial` if the writer recorded errors,
and the critic's scores are recorded in `critiques` without changing the status. The
single-prompt baseline is no longer part of the graph: the evals call it directly.
"""

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from brandforge.agents.critic import critique_variants
from brandforge.agents.planner import plan_brief
from brandforge.agents.writer import write_variants
from brandforge.models import BrandProfile, Brief, RunState, new_run_state

# Same shape as the agents: read state, return a partial state update.
Node = Callable[[RunState], dict[str, Any]]

PLANNER_NODE = "planner"
WRITER_NODE = "writer"
CRITIC_NODE = "critic"


def build_graph(
    plan: Node | None = None,
    write: Node | None = None,
    critique: Node | None = None,
) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    """Compile the graph. `plan`, `write` and `critique` default to the real agents, looked up
    at call time so tests can replace them."""
    planner = plan or plan_brief
    writer = write or write_variants
    critic = critique or critique_variants

    def planner_node(state: RunState) -> dict[str, Any]:
        return planner(state)

    def writer_node(state: RunState) -> dict[str, Any]:
        update = writer(state)
        return {**update, "status": "partial" if update.get("errors") else "complete"}

    def critic_node(state: RunState) -> dict[str, Any]:
        return critic(state)

    builder = StateGraph(RunState)
    builder.add_node(PLANNER_NODE, planner_node)
    builder.add_node(WRITER_NODE, writer_node)
    builder.add_node(CRITIC_NODE, critic_node)
    builder.add_edge(START, PLANNER_NODE)
    builder.add_edge(PLANNER_NODE, WRITER_NODE)
    builder.add_edge(WRITER_NODE, CRITIC_NODE)
    builder.add_edge(CRITIC_NODE, END)
    return builder.compile()


def run_graph(
    brief: Brief,
    brand: BrandProfile,
    *,
    plan: Node | None = None,
    write: Node | None = None,
    critique: Node | None = None,
    run_id: str | None = None,
) -> RunState:
    """Run one brief through the graph and return the final state."""
    graph = build_graph(plan, write, critique)
    return cast(RunState, graph.invoke(new_run_state(brief, brand, run_id=run_id)))
