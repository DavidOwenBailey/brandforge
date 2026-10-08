"""Checkpointing: state is saved after every node, in a SQLite file, and reads back as models."""

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from brandforge.brands import load_brand
from brandforge.checkpointing import RunStep, _contract_types, load_run_steps, open_checkpointer
from brandforge.graph import run_graph
from brandforge.models import (
    BrandProfile,
    Brief,
    Critique,
    Example,
    Plan,
    RunError,
    RunResult,
    RunState,
    Usage,
    Variant,
    VariantResult,
)


@pytest.fixture
def brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "state" / "checkpoints.sqlite"


def _plan(state: RunState) -> dict[str, Any]:
    plan = Plan(
        audience="commuters",
        angle="save time",
        channels=list(state["brief"].channels),
        variants_per_channel=1,
    )
    return {"plan": plan, "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.002)}


def _write(state: RunState) -> dict[str, Any]:
    variant = Variant(id="search-1", channel="search", headline="Ride", body="Now", cta="Go")
    return {"variants": [variant], "usage": Usage(input_tokens=100, output_tokens=50)}


def _critique(passed: bool) -> Callable[[RunState], dict[str, Any]]:
    def critique(state: RunState) -> dict[str, Any]:
        critiques = [
            Critique(
                variant_id=variant.id,
                scores={"voice": 5 if passed else 2},
                overall=5.0 if passed else 2.0,
                passed=passed,
                fixes=[] if passed else ["Tighten the headline"],
            )
            for variant in state["variants"]
        ]
        return {"critiques": critiques, "usage": Usage()}

    return critique


class RevisionScript:
    """Fails the first critique and passes the second; the reviser counts a revision."""

    def __init__(self) -> None:
        self.critic_calls = 0

    def critique(self, state: RunState) -> dict[str, Any]:
        self.critic_calls += 1
        return _critique(self.critic_calls > 1)(state)

    def revise(self, state: RunState) -> dict[str, Any]:
        return {"critiques": [], "revision_count": state["revision_count"] + 1, "usage": Usage()}


def _run_and_read(brief: Brief, db: Path, run_id: str, **nodes: Any) -> list[RunStep]:
    """Run the graph with checkpointing on, then read the run back through a new connection."""
    with open_checkpointer(db) as saver:
        run_graph(brief, load_brand("voltride"), run_id=run_id, checkpointer=saver, **nodes)
    with open_checkpointer(db, create=False) as saver:
        return load_run_steps(saver, run_id)


def test_state_is_saved_after_every_node_and_read_back_in_order(brief: Brief, db: Path) -> None:
    steps = _run_and_read(brief, db, "run-1", plan=_plan, write=_write, critique=_critique(True))

    assert [step.node for step in steps] == [None, "planner", "writer", "critic", "assembler"]
    assert [step.step for step in steps] == [0, 1, 2, 3, 4]
    assert [step.next_nodes for step in steps] == [
        ("planner",),
        ("writer",),
        ("critic",),
        ("assembler",),
        (),
    ]
    assert [step.state["status"] for step in steps] == ["running"] * 4 + ["complete"]
    assert [len(step.state["variants"]) for step in steps] == [0, 0, 1, 1, 1]
    assert [step.state["usage"].total_tokens for step in steps] == [0, 15, 165, 165, 165]
    assert steps[-1].state["result"] is not None
    assert steps[0].created_at is not None


def test_checkpointed_state_comes_back_as_models(brief: Brief, db: Path) -> None:
    steps = _run_and_read(brief, db, "run-1", plan=_plan, write=_write, critique=_critique(True))

    state = steps[-1].state
    assert isinstance(state["brief"], Brief)
    assert isinstance(state["brand"], BrandProfile)
    assert isinstance(state["plan"], Plan)
    assert isinstance(state["variants"][0], Variant)
    assert isinstance(state["critiques"][0], Critique)
    assert isinstance(state["usage"], Usage)
    assert isinstance(state["result"], RunResult)
    assert state["brand"] == load_brand("voltride")


def test_the_allowlist_covers_every_model_a_checkpoint_can_hold() -> None:
    """LangGraph will block types it has not been told to trust, so a strict serializer given our
    allowlist must still rebuild each contract; without it the same serializer cannot."""
    strict = JsonPlusSerializer(allowed_msgpack_modules=None)
    trusted = strict.with_msgpack_allowlist(_contract_types())
    brand = load_brand("voltride")
    variant = Variant(id="v", channel="search", headline="H", body="B", cta="C")
    variant_result = VariantResult(variant=variant, critique=None, flagged=True)
    contracts: list[BaseModel] = [
        brand,
        brand.rubric,
        brand.rubric.criteria[0],
        Brief(product="p", audience="a", objective="awareness", channels=["search"]),
        Plan(audience="a", angle="x", channels=["search"], variants_per_channel=1),
        Example(brand_id="voltride", channel="search", headline="H", body="B", cta="C"),
        variant,
        Critique(variant_id="v", scores={"voice": 4}, overall=4.0, passed=True),
        RunError(node="writer", message="boom", fatal=True),
        Usage(input_tokens=1, output_tokens=2, cost_usd=0.5),
    ]
    contracts += [
        variant_result,
        RunResult(
            run_id="r",
            status="partial",
            brand_id="voltride",
            brand_version="1",
            rubric_version="1",
            variants=[variant_result],
            revision_count=0,
            errors=[],
            usage=Usage(),
        ),
    ]

    for contract in contracts:
        typed = trusted.dumps_typed(contract)
        assert trusted.loads_typed(typed) == contract, type(contract).__name__
        assert strict.loads_typed(typed) != contract, type(contract).__name__


def test_each_revision_pass_gets_its_own_checkpoints(brief: Brief, db: Path) -> None:
    script = RevisionScript()

    steps = _run_and_read(
        brief, db, "run-1", plan=_plan, write=_write, critique=script.critique, revise=script.revise
    )

    assert [step.node for step in steps] == [
        None,
        "planner",
        "writer",
        "critic",
        "reviser",
        "critic",
        "assembler",
    ]
    assert [step.state["revision_count"] for step in steps] == [0, 0, 0, 0, 1, 1, 1]


def test_a_node_that_fails_leaves_its_error_in_the_checkpoints(brief: Brief, db: Path) -> None:
    def broken_write(state: RunState) -> dict[str, Any]:
        raise RuntimeError("provider down")

    steps = _run_and_read(brief, db, "run-1", plan=_plan, write=broken_write)

    assert [step.node for step in steps] == [None, "planner", "writer", "assembler"]
    last = steps[-1].state
    assert last["status"] == "failed"
    assert [(e.node, e.fatal) for e in last["errors"]] == [("writer", True)]
    assert "provider down" in last["errors"][0].message


def test_a_run_cut_short_can_still_be_read_up_to_the_last_node(brief: Brief, db: Path) -> None:
    def interrupted(state: RunState) -> dict[str, Any]:
        raise KeyboardInterrupt  # not an Exception, so the node guard lets it stop the run

    with open_checkpointer(db) as saver, pytest.raises(KeyboardInterrupt):
        run_graph(
            brief,
            load_brand("voltride"),
            plan=_plan,
            write=_write,
            critique=interrupted,
            run_id="run-1",
            checkpointer=saver,
        )
    with open_checkpointer(db, create=False) as saver:
        steps = load_run_steps(saver, "run-1")

    assert [step.node for step in steps] == [None, "planner", "writer"]
    assert steps[-1].next_nodes == ("critic",)
    assert steps[-1].state["status"] == "running"
    assert len(steps[-1].state["variants"]) == 1


def test_runs_are_kept_apart_by_run_id(brief: Brief, db: Path) -> None:
    _run_and_read(brief, db, "run-1", plan=_plan, write=_write, critique=_critique(True))
    steps = _run_and_read(brief, db, "run-2", plan=_plan, write=_write, critique=_critique(True))

    assert {step.state["run_id"] for step in steps} == {"run-2"}
    assert steps[-1].state["usage"].total_tokens == 165  # run-1's usage is not carried over


def test_an_unknown_run_has_no_steps(brief: Brief, db: Path) -> None:
    _run_and_read(brief, db, "run-1", plan=_plan, write=_write, critique=_critique(True))

    with open_checkpointer(db, create=False) as saver:
        assert load_run_steps(saver, "no-such-run") == []


def test_run_graph_refuses_a_run_id_that_already_has_checkpoints(brief: Brief, db: Path) -> None:
    _run_and_read(brief, db, "run-1", plan=_plan, write=_write, critique=_critique(True))

    with open_checkpointer(db) as saver, pytest.raises(ValueError, match="already has checkpoints"):
        run_graph(
            brief,
            load_brand("voltride"),
            plan=_plan,
            write=_write,
            critique=_critique(True),
            run_id="run-1",
            checkpointer=saver,
        )


def test_without_a_checkpointer_run_graph_writes_nothing(
    brief: Brief, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    run_graph(brief, load_brand("voltride"), plan=_plan, write=_write, critique=_critique(True))

    assert list(tmp_path.iterdir()) == []


def test_open_checkpointer_creates_the_folder_and_file(db: Path) -> None:
    assert not db.parent.exists()

    with open_checkpointer(db):
        pass

    assert db.is_file()


def test_reading_a_missing_database_does_not_create_one(db: Path) -> None:
    with (
        pytest.raises(FileNotFoundError, match="no checkpoint database"),
        open_checkpointer(db, create=False),
    ):
        pass

    assert not db.parent.exists()


def test_the_connection_is_closed_when_the_block_ends(db: Path) -> None:
    with open_checkpointer(db) as saver:
        conn = saver.conn  # type: ignore[attr-defined]

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("select 1")
