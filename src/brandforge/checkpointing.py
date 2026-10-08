"""Checkpointing (BF-24): save the run state after every node, and read it back to inspect a run.

A graph compiled with a checkpointer writes the whole state to it after each node. Each run is
one LangGraph "thread" whose ID is the run's own `run_id`, so `brandforge inspect <run_id>` can
find it. The store is a local SQLite file (`settings.checkpoint_db`), per the local-first
decision (ADR 0007). Why it is built this way is in ADR 0018.

`open_checkpointer` gives the saver to pass to `run_graph`. `load_run_steps` reads one run back
as a list of steps, oldest first, each holding the state as it was after a node.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from brandforge import models
from brandforge.graph import build_graph
from brandforge.models import RunState


def _contract_types() -> list[tuple[str, str]]:
    """Every Pydantic model in `brandforge.models`, as the (module, class) pairs LangGraph's
    allowlist wants.

    State holds these models directly, and LangGraph only rebuilds a type it has been told to
    trust. Taking the list from the module means a new contract is allowed without anyone having
    to remember to add it here.
    """
    return [
        (models.__name__, name)
        for name, obj in vars(models).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == models.__name__
    ]


@contextmanager
def open_checkpointer(path: Path, *, create: bool = True) -> Iterator[BaseCheckpointSaver[str]]:
    """Open the SQLite checkpoint store at `path` and close it when the block ends.

    With `create` (the default) the file and its folder are made if they are missing, which is
    what a run needs. Reading a run back passes `create=False`, so asking about a run never
    leaves an empty database behind; a missing file raises `FileNotFoundError`.
    """
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise FileNotFoundError(f"no checkpoint database at {path}")
    # check_same_thread=False is what SqliteSaver.from_conn_string does too: LangGraph may run
    # a node on another thread, and the saver guards the connection with its own lock.
    with closing(sqlite3.connect(path, check_same_thread=False)) as conn:
        serde = JsonPlusSerializer(allowed_msgpack_modules=_contract_types())
        yield SqliteSaver(conn, serde=serde)


def thread_config(run_id: str) -> RunnableConfig:
    """The config that points LangGraph at one run's checkpoints."""
    return {"configurable": {"thread_id": run_id}}


@dataclass(frozen=True)
class RunStep:
    """The state of a run at one checkpoint.

    `node` is the node that ran to produce this state, or None for the initial state before any
    node ran. `next_nodes` is what was due to run next: empty once the run has finished, and
    non-empty on the last step of a run that was cut short.
    """

    step: int
    node: str | None
    next_nodes: tuple[str, ...]
    created_at: str | None
    state: RunState


def load_run_steps(saver: BaseCheckpointSaver[str], run_id: str) -> list[RunStep]:
    """Every checkpoint of a run, oldest first. An unknown run ID gives an empty list.

    Checkpoints do not record which node produced them, so it is read from the one before: the
    node that was due to run next there is the one that ran. LangGraph's own first checkpoint
    only holds the raw input, with no state to show, so it is left out.
    """
    graph = build_graph(checkpointer=saver)
    numbered = [
        (step, snapshot)
        for snapshot in graph.get_state_history(thread_config(run_id))
        if (step := (snapshot.metadata or {}).get("step", -1)) >= 0
    ]
    numbered.sort(key=lambda pair: pair[0])

    steps: list[RunStep] = []
    due: tuple[str, ...] = ()
    for step, snapshot in numbered:
        steps.append(
            RunStep(
                step=step,
                node=", ".join(due) or None,
                next_nodes=tuple(snapshot.next),
                created_at=snapshot.created_at,
                state=cast(RunState, snapshot.values),
            )
        )
        due = tuple(snapshot.next)
    return steps
