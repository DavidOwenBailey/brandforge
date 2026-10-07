"""The LangGraph pipeline: nodes, edges and the entry point the CLI and evals call.

The graph runs the planner (BF-13), the writer (BF-14) and the brand critic (BF-15). After the
critic, the router (BF-16) decides: `revise` goes to the reviser (BF-17) and back to the critic,
at most `budgets.max_revisions` times; `assemble` and `stop` both go to the assembler (BF-18),
which packages the `RunResult` and sets the final status. No other node touches `status`, so it
stays `running` until the assembler runs. The single-prompt baseline is no longer part of the
graph: the evals call it directly.
"""

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from brandforge.agents.assembler import assemble_result
from brandforge.agents.critic import critique_variants
from brandforge.agents.planner import plan_brief
from brandforge.agents.reviser import revise_variants
from brandforge.agents.writer import write_variants
from brandforge.models import BrandProfile, Brief, RunState, new_run_state
from brandforge.router import RouteAction, route

# Same shape as the agents: read state, return a partial state update.
Node = Callable[[RunState], dict[str, Any]]

PLANNER_NODE = "planner"
WRITER_NODE = "writer"
CRITIC_NODE = "critic"
REVISER_NODE = "reviser"
ASSEMBLER_NODE = "assembler"


def _after_critic(state: RunState) -> RouteAction:
    return route(state)


def build_graph(
    plan: Node | None = None,
    write: Node | None = None,
    critique: Node | None = None,
    revise: Node | None = None,
    assemble: Node | None = None,
) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    """Compile the graph. `plan`, `write`, `critique`, `revise` and `assemble` default to the
    real agents, looked up at call time so tests can replace them."""
    planner = plan or plan_brief
    writer = write or write_variants
    critic = critique or critique_variants
    reviser = revise or revise_variants
    assembler = assemble or assemble_result

    def planner_node(state: RunState) -> dict[str, Any]:
        return planner(state)

    def writer_node(state: RunState) -> dict[str, Any]:
        return writer(state)

    def critic_node(state: RunState) -> dict[str, Any]:
        return critic(state)

    def reviser_node(state: RunState) -> dict[str, Any]:
        return reviser(state)

    def assembler_node(state: RunState) -> dict[str, Any]:
        return assembler(state)

    builder = StateGraph(RunState)
    builder.add_node(PLANNER_NODE, planner_node)
    builder.add_node(WRITER_NODE, writer_node)
    builder.add_node(CRITIC_NODE, critic_node)
    builder.add_node(REVISER_NODE, reviser_node)
    builder.add_node(ASSEMBLER_NODE, assembler_node)
    builder.add_edge(START, PLANNER_NODE)
    builder.add_edge(PLANNER_NODE, WRITER_NODE)
    builder.add_edge(WRITER_NODE, CRITIC_NODE)
    builder.add_conditional_edges(
        CRITIC_NODE,
        _after_critic,
        {"revise": REVISER_NODE, "assemble": ASSEMBLER_NODE, "stop": ASSEMBLER_NODE},
    )
    builder.add_edge(REVISER_NODE, CRITIC_NODE)
    builder.add_edge(ASSEMBLER_NODE, END)
    return builder.compile()


def run_graph(
    brief: Brief,
    brand: BrandProfile,
    *,
    plan: Node | None = None,
    write: Node | None = None,
    critique: Node | None = None,
    revise: Node | None = None,
    assemble: Node | None = None,
    run_id: str | None = None,
) -> RunState:
    """Run one brief through the graph and return the final state."""
    graph = build_graph(plan, write, critique, revise, assemble)
    return cast(RunState, graph.invoke(new_run_state(brief, brand, run_id=run_id)))
