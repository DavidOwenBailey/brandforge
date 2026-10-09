"""The LangGraph pipeline: nodes, edges and the entry point the CLI and evals call.

The graph runs the planner (BF-13), the retriever (BF-31), the writer (BF-14) and the brand
critic (BF-15). The retriever writes the nearest approved examples for each channel into state.
An empty index leaves that list empty, logs a warning and the run continues (ADR 0024).
With `retrieval_enabled` off the node still runs, writes an empty list and does not search
(BF-32, ADR 0025). After the critic, the router (BF-16) decides: `revise` goes to the
reviser (BF-17) and back to the critic, at most `budgets.max_revisions` times; `assemble`
and `stop` both go to the assembler (BF-18), which packages the `RunResult` and sets the
final status. No other node touches
`status`, so it stays `running` until the assembler runs. The single-prompt baseline is no
longer part of the graph: the evals call it directly.

Error edges (BF-22): the planner, retriever, writer, critic and reviser each run inside a guard.
If one raises after the gateway's retries, the guard records a fatal `RunError` (and the usage of
any failed calls) instead of letting the exception out, and the edge after that node sends the
run to the assembler with whatever state exists. The assembler then ends it `partial` or `failed`,
so a run always returns a result. An empty retrieval is not a failure and does not take this
edge. The assembler is not guarded: it is plain code over state.

Run budget (BF-23): the same guard installs the run's token and wall-clock budget around each
node (`brandforge.budget`), which is what the gateway checks before every model call. A node
the budget stops raises `BudgetExceededError`, which takes the error edge like any other
failure. The router checks the same limits before starting another revision.

Checkpointing (BF-24): `build_graph` and `run_graph` take an optional checkpointer. With one, the
whole state is saved after every node, keyed by `run_id`, so a run can be read back later with
`brandforge.checkpointing`. Without one nothing is written, which is what the evals and most
tests want.

Tracing (BF-25): `run_graph` opens one Langfuse trace for the run and every node runs inside its
own span (`_traced`), with the gateway's model calls nested under it. The trace ID is derived
from the `run_id`, stored in state, and ends up in the `RunResult`. A node that the guard turned
into a fatal `RunError` shows as an error span, even though no exception left it. See
`brandforge.tracing` and ADR 0019. With tracing off none of this does anything.

Logging (BF-26): `run_graph` binds `run_id` (and `trace_id`, when there is one) for the whole
invocation, and logs `run_started` / `run_finished`. Every other log emitted during the run —
the guard, the router, the gateway — picks up the same `run_id`. See `brandforge.logging` and
ADR 0020. The lines are JSON on stderr once the CLI has configured logging; they are not part of
the printed result.

Cost by node (BF-27): the usage a node returns is attributed to that node's name before it is
reduced into the run total, including the usage a failed node still spent. A node that runs
again (the critic, after a revision) is summed into the same row.
"""

import logging
from collections.abc import Callable
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from brandforge import tracing
from brandforge.agents.assembler import assemble_result
from brandforge.agents.critic import critique_variants
from brandforge.agents.planner import plan_brief
from brandforge.agents.retriever import retrieve_examples
from brandforge.agents.reviser import revise_variants
from brandforge.agents.writer import write_variants
from brandforge.budget import RunBudget, active_budget
from brandforge.config import Settings, get_settings
from brandforge.llm.base import BudgetExceededError, StructuredOutputError
from brandforge.logging import bind_run, get_logger
from brandforge.models import (
    BrandProfile,
    Brief,
    RunError,
    RunResult,
    RunState,
    Usage,
    new_run_state,
)
from brandforge.router import RouteAction, route

logger = logging.getLogger(__name__)
slog = get_logger(__name__)

# Same shape as the agents: read state, return a partial state update.
Node = Callable[[RunState], dict[str, Any]]

PLANNER_NODE = "planner"
RETRIEVER_NODE = "retriever"
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
    Calls that were made before the failure still cost money, so the usage of a
    `StructuredOutputError` or a `BudgetExceededError` is kept. Only `Exception` is caught:
    `KeyboardInterrupt` and `SystemExit` still stop the run.

    The node runs under a `RunBudget` made from the state it was given, so the gateway can stop
    it from spending past the run's token or wall-clock limit (BF-23).
    """

    def run(state: RunState) -> dict[str, Any]:
        try:
            budget = RunBudget.for_state(state, get_settings().budgets)
            with active_budget(budget):
                return node(state)
        except Exception as exc:
            logger.warning("Node %r failed; sending the run to the assembler: %s", name, exc)
            update: dict[str, Any] = {
                "errors": [RunError(node=name, message=_describe(exc), fatal=True)]
            }
            if isinstance(exc, StructuredOutputError | BudgetExceededError):
                update["usage"] = exc.usage
            return update

    return run


def _prompt_version(settings: Settings, name: str) -> str | None:
    """The prompt file a node uses, like `planner_v2`, or `None` for a node with no prompt."""
    version: str | None = getattr(settings, f"{name}_prompt_version", None)
    return f"{name}_{version}" if version else None


def _summarise(update: dict[str, Any]) -> dict[str, Any]:
    """A node's state update as a small, readable dict for its span: counts, not contents.

    The contents (variants, critiques) are already in the model calls' generations and in the
    checkpoints, so repeating them on every span would only make the trace long.
    """
    summary: dict[str, Any] = {}
    for key, value in update.items():
        if isinstance(value, Usage):
            summary["tokens"] = value.total_tokens
            summary["cost_usd"] = value.cost_usd
        elif isinstance(value, RunResult):
            summary["flagged"] = value.flagged_count
        elif isinstance(value, list):
            summary[key] = len(value)
        elif isinstance(value, str | int | float | bool) or value is None:
            summary[key] = value
        else:
            summary[key] = type(value).__name__
    return summary


def _traced(name: str, node: Node) -> Node:
    """Wrap a node so it runs inside its own trace span, outside the guard.

    The guard turns a failure into a fatal `RunError` instead of an exception, so the span would
    never see one. The span is marked as an error from that `RunError` instead, which is what makes
    a failed node stand out in the trace. An exception from an unguarded node (the assembler)
    still passes through and marks the span itself.
    """

    def run(state: RunState) -> dict[str, Any]:
        version = _prompt_version(get_settings(), name)
        with tracing.node_span(name, state, version=version) as span:
            update = node(state)
            usage = update.get("usage")
            if isinstance(usage, Usage):
                update["usage"] = usage.attributed_to(name)
            span.update(output=_summarise(update))
            fatal = [error for error in update.get("errors", []) if error.fatal]
            if fatal:
                span.update(level="ERROR", status_message=fatal[0].message)
            return update

    return run


def _halted(state: RunState) -> bool:
    return any(error.fatal for error in state["errors"])


def _continue_or_assemble(state: RunState) -> Literal["continue", "assemble"]:
    """The edge after the planner, retriever, writer and reviser: carry on unless a node failed."""
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
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    retrieve: Node | None = None,
) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    """Compile the graph. `plan`, `retrieve`, `write`, `critique`, `revise` and `assemble`
    default to the real agents, looked up at call time so tests can replace them. With a
    `checkpointer`, state is saved after every node (BF-24)."""
    planner = _traced(PLANNER_NODE, _guarded(PLANNER_NODE, plan or plan_brief))
    retriever = _traced(RETRIEVER_NODE, _guarded(RETRIEVER_NODE, retrieve or retrieve_examples))
    writer = _traced(WRITER_NODE, _guarded(WRITER_NODE, write or write_variants))
    critic = _traced(CRITIC_NODE, _guarded(CRITIC_NODE, critique or critique_variants))
    reviser = _traced(REVISER_NODE, _guarded(REVISER_NODE, revise or revise_variants))
    assembler = _traced(ASSEMBLER_NODE, assemble or assemble_result)

    def planner_node(state: RunState) -> dict[str, Any]:
        return planner(state)

    def retriever_node(state: RunState) -> dict[str, Any]:
        return retriever(state)

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
    builder.add_node(RETRIEVER_NODE, retriever_node)
    builder.add_node(WRITER_NODE, writer_node)
    builder.add_node(CRITIC_NODE, critic_node)
    builder.add_node(REVISER_NODE, reviser_node)
    builder.add_node(ASSEMBLER_NODE, assembler_node)
    builder.add_edge(START, PLANNER_NODE)
    builder.add_conditional_edges(
        PLANNER_NODE,
        _continue_or_assemble,
        {"continue": RETRIEVER_NODE, "assemble": ASSEMBLER_NODE},
    )
    builder.add_conditional_edges(
        RETRIEVER_NODE,
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
    return builder.compile(checkpointer=checkpointer)


def run_graph(
    brief: Brief,
    brand: BrandProfile,
    *,
    plan: Node | None = None,
    write: Node | None = None,
    critique: Node | None = None,
    revise: Node | None = None,
    assemble: Node | None = None,
    retrieve: Node | None = None,
    run_id: str | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> RunState:
    """Run one brief through the graph and return the final state.

    With a `checkpointer`, the state is saved after every node under the run's `run_id` (BF-24),
    so the run can be inspected afterwards, including one that ended early. A `run_id` that
    already has checkpoints is refused: a second run on the same thread would start from the
    first one's final state, and the usage and error reducers would add to its totals.

    With tracing on (BF-25) the whole run is one Langfuse trace, flushed before this returns. Its
    ID is in `state["trace_id"]` and the result's `trace_id`; both are `None` with tracing off.

    The run's `run_id` is bound onto the logs for the whole call (BF-26), including when the run
    raises, and a `run_started` / `run_finished` pair is written around it.
    """
    graph = build_graph(
        plan, write, critique, revise, assemble, checkpointer=checkpointer, retrieve=retrieve
    )
    settings = get_settings()
    state = new_run_state(brief, brand, run_id=run_id)
    state["trace_id"] = tracing.trace_id_for(state["run_id"], settings)
    config: RunnableConfig = {"configurable": {"thread_id": state["run_id"]}}
    if checkpointer is not None and checkpointer.get_tuple(config) is not None:
        raise ValueError(f"run_id {state['run_id']!r} already has checkpoints; use a new run_id")

    final: RunState | None = None
    with bind_run(state["run_id"], trace_id=state["trace_id"]):
        slog.info("run_started", brand_id=brand.id)
        try:
            with tracing.run_trace(state, settings) as root:
                final = cast(RunState, graph.invoke(state, config))
                result = final["result"]
                if result is not None:
                    root.update(
                        output={
                            "status": result.status,
                            "variants": len(result.variants),
                            "flagged": result.flagged_count,
                            "revisions": result.revision_count,
                            "tokens": result.usage.total_tokens,
                            "cost_usd": result.usage.cost_usd,
                        }
                    )
        finally:
            finished = None if final is None else final["result"]
            slog.info(
                "run_finished",
                status=None if finished is None else finished.status,
                tokens=None if finished is None else finished.usage.total_tokens,
                cost_usd=None if finished is None else finished.usage.cost_usd,
            )
    if final is None:
        # `invoke` either returns a state or raises, and a raise leaves this function in the
        # `finally` above. This is here so the type checker can see that we always return one.
        raise RuntimeError("the graph returned no state")
    return final
