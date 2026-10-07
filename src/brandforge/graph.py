"""The LangGraph pipeline: nodes, edges and the entry point the CLI and evals call.

The graph runs the planner (BF-13), the writer (BF-14) and the brand critic (BF-15). After the
critic, the router (BF-16) decides: `revise` goes to the reviser (BF-17) and back to the critic,
at most `budgets.max_revisions` times; `assemble` and `stop` both go to the assembler (BF-18),
which packages the `RunResult` and sets the final status. No other node touches `status`, so it
stays `running` until the assembler runs. The single-prompt baseline is no longer part of the
graph: the evals call it directly.

Error edges (BF-22): the planner, writer, critic and reviser each run inside a guard. If one
raises after the gateway's retries, the guard records a fatal `RunError` (and the usage of any
failed calls) instead of letting the exception out, and the edge after that node sends the run to
the assembler with whatever state exists. The assembler then ends it `partial` or `failed`, so a
run always returns a result. The assembler is not guarded: it is plain code over state.
"""

import logging
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from brandforge.agents.assembler import assemble_result
from brandforge.agents.critic import critique_variants
from brandforge.agents.planner import plan_brief
from brandforge.agents.reviser import revise_variants
from brandforge.agents.writer import write_variants
from brandforge.llm.base import StructuredOutputError
from brandforge.models import BrandProfile, Brief, RunError, RunState, new_run_state
from brandforge.router import RouteAction, route

logger = logging.getLogger(__name__)

# Same shape as the agents: read state, return a partial state update.
Node = Callable[[RunState], dict[str, Any]]

PLANNER_NODE = "planner"
WRITER_NODE = "writer"
CRITIC_NODE = "critic"
REVISER_NODE = "reviser"
ASSEMBLER_NODE = "assembler"


def _describe(exc: Exception) -> str:
    """The error as one non-empty line: its type, then its message when it has one."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _guarded(name: str, node: Node) -> Node:
    """Wrap a node so that an exception becomes a fatal `RunError` in state, never a crash.

    The failed node returns no other keys, so state is exactly what it was before the node ran.
    Calls that were made before the failure still cost money, so a `StructuredOutputError`'s usage
    is kept. Only `Exception` is caught: `KeyboardInterrupt` and `SystemExit` still stop the run.
    """

    def run(state: RunState) -> dict[str, Any]:
        try:
            return node(state)
        except Exception as exc:
            logger.warning("Node %r failed; sending the run to the assembler: %s", name, exc)
            update: dict[str, Any] = {
                "errors": [RunError(node=name, message=_describe(exc), fatal=True)]
            }
            if isinstance(exc, StructuredOutputError):
                update["usage"] = exc.usage
            return update

    return run


def _halted(state: RunState) -> bool:
    return any(error.fatal for error in state["errors"])


def _continue_or_assemble(state: RunState) -> Literal["continue", "assemble"]:
    """The edge after the planner, writer and reviser: carry on unless a node failed."""
    return "assemble" if _halted(state) else "continue"


def _after_critic(state: RunState) -> RouteAction:
    """A failed critic goes straight to the assembler. Otherwise the router decides."""
    return "stop" if _halted(state) else route(state)


def build_graph(
    plan: Node | None = None,
    write: Node | None = None,
    critique: Node | None = None,
    revise: Node | None = None,
    assemble: Node | None = None,
) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    """Compile the graph. `plan`, `write`, `critique`, `revise` and `assemble` default to the
    real agents, looked up at call time so tests can replace them."""
    planner = _guarded(PLANNER_NODE, plan or plan_brief)
    writer = _guarded(WRITER_NODE, write or write_variants)
    critic = _guarded(CRITIC_NODE, critique or critique_variants)
    reviser = _guarded(REVISER_NODE, revise or revise_variants)
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
    builder.add_conditional_edges(
        PLANNER_NODE,
        _continue_or_assemble,
        {"continue": WRITER_NODE, "assemble": ASSEMBLER_NODE},
    )
    builder.add_conditional_edges(
        WRITER_NODE,
        _continue_or_assemble,
        {"continue": CRITIC_NODE, "assemble": ASSEMBLER_NODE},
    )
    builder.add_conditional_edges(
        CRITIC_NODE,
        _after_critic,
        {"revise": REVISER_NODE, "assemble": ASSEMBLER_NODE, "stop": ASSEMBLER_NODE},
    )
    builder.add_conditional_edges(
        REVISER_NODE,
        _continue_or_assemble,
        {"continue": CRITIC_NODE, "assemble": ASSEMBLER_NODE},
    )
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
