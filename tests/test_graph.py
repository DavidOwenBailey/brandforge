"""The graph: planner, writer, critic and the revision loop, with typed state, reducers and
error edges."""

from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from brandforge.brands import load_brand
from brandforge.config import Settings
from brandforge.graph import (
    ASSEMBLER_NODE,
    CRITIC_NODE,
    PLANNER_NODE,
    REVISER_NODE,
    WRITER_NODE,
    build_graph,
    run_graph,
)
from brandforge.llm.base import GatewayError, StructuredOutputError
from brandforge.models import (
    Brief,
    Critique,
    Plan,
    RunError,
    RunState,
    Usage,
    Variant,
    new_run_state,
)


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


@pytest.fixture(autouse=True)
def isolated_router_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The router inside the graph reads the revision cap from settings; pin the defaults."""
    monkeypatch.setattr("brandforge.router.get_settings", lambda: IsolatedSettings())


@pytest.fixture
def brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )


class FakePlan:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        if self.error:
            raise self.error
        plan = Plan(
            audience="commuters",
            angle="save time",
            channels=list(state["brief"].channels),
            variants_per_channel=2,
        )
        return {"plan": plan, "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.002)}


class FakeWrite:
    def __init__(
        self, error: Exception | None = None, errors: list[RunError] | None = None
    ) -> None:
        self.error = error
        self.errors = errors or []
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        if self.error:
            raise self.error
        variant = Variant(id="search-1", channel="search", headline="H", body="B", cta="C")
        update: dict[str, Any] = {
            "variants": [variant],
            "usage": Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        }
        if self.errors:
            update["errors"] = self.errors
        return update


class FakeCritique:
    """Scores every variant 5 on one criterion. Usage is zero unless a test gives some."""

    def __init__(self, usage: Usage | None = None, error: Exception | None = None) -> None:
        self.usage = usage or Usage()
        self.error = error
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        if self.error:
            raise self.error
        critiques = [
            Critique(
                variant_id=variant.id,
                scores={"voice": 5},
                overall=5.0,
                passed=True,
            )
            for variant in state["variants"]
        ]
        return {"critiques": critiques, "usage": self.usage}


class ScriptedCritique:
    """Passes or fails every variant according to `passes`, one entry per call.

    The last entry repeats once the script runs out.
    """

    def __init__(self, passes: list[bool], usage: Usage | None = None) -> None:
        self.passes = list(passes)
        self.usage = usage or Usage()
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        passed = self.passes[min(len(self.calls), len(self.passes)) - 1]
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
        return {"critiques": critiques, "usage": self.usage}


class FakeRevise:
    """Rewrites every variant's headline, counts one revision and drops the stale critiques."""

    def __init__(self, usage: Usage | None = None, error: Exception | None = None) -> None:
        self.usage = usage or Usage()
        self.error = error
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        if self.error:
            raise self.error
        round_number = state["revision_count"] + 1
        variants = [
            variant.model_copy(update={"headline": f"{variant.headline}-r{round_number}"})
            for variant in state["variants"]
        ]
        return {
            "variants": variants,
            "critiques": [],
            "revision_count": round_number,
            "usage": self.usage,
        }


class FakeAssemble:
    """Records the state it is given and returns a fixed status update."""

    def __init__(self) -> None:
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        return {"status": "failed"}


def test_graph_runs_the_planner_writer_and_critic_then_loops_through_the_reviser() -> None:
    graph = build_graph(FakePlan(), FakeWrite(), FakeCritique(), FakeRevise())

    nodes = graph.get_graph()
    assert set(nodes.nodes) == {
        "__start__",
        PLANNER_NODE,
        WRITER_NODE,
        CRITIC_NODE,
        REVISER_NODE,
        ASSEMBLER_NODE,
        "__end__",
    }
    edges = {(e.source, e.target) for e in nodes.edges}
    assert edges == {
        ("__start__", PLANNER_NODE),
        (PLANNER_NODE, WRITER_NODE),
        (PLANNER_NODE, ASSEMBLER_NODE),
        (WRITER_NODE, CRITIC_NODE),
        (WRITER_NODE, ASSEMBLER_NODE),
        (CRITIC_NODE, REVISER_NODE),
        (CRITIC_NODE, ASSEMBLER_NODE),
        (REVISER_NODE, CRITIC_NODE),
        (REVISER_NODE, ASSEMBLER_NODE),
        (ASSEMBLER_NODE, "__end__"),
    }


def test_run_graph_returns_variants_critiques_and_complete_status(brief: Brief) -> None:
    write = FakeWrite()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=write,
        critique=FakeCritique(),
        run_id="run-1",
    )

    assert state["run_id"] == "run-1"
    assert state["status"] == "complete"
    assert [v.id for v in state["variants"]] == ["search-1"]
    assert [c.variant_id for c in state["critiques"]] == ["search-1"]
    assert state["errors"] == []
    assert len(write.calls) == 1
    assert state["result"] is not None
    assert state["result"].run_id == "run-1"
    assert state["result"].status == "complete"
    assert [item.variant.id for item in state["result"].variants] == ["search-1"]


def test_writer_receives_the_plan_the_planner_wrote(brief: Brief) -> None:
    write = FakeWrite()

    state = run_graph(
        brief, load_brand("voltride"), plan=FakePlan(), write=write, critique=FakeCritique()
    )

    assert state["plan"] is not None
    assert state["plan"].angle == "save time"
    assert write.calls[0]["plan"] == state["plan"]


def test_critic_receives_the_variants_the_writer_wrote(brief: Brief) -> None:
    critique = FakeCritique()

    run_graph(brief, load_brand("voltride"), plan=FakePlan(), write=FakeWrite(), critique=critique)

    assert [v.id for v in critique.calls[0]["variants"]] == ["search-1"]
    assert critique.calls[0]["brand"].id == "voltride"


def test_writer_errors_are_kept_and_mark_the_run_partial(brief: Brief) -> None:
    error = RunError(node="writer", message="channel 'search': asked for 2 variants, got 1")

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(errors=[error]),
        critique=FakeCritique(),
    )

    assert state["status"] == "partial"
    assert state["errors"] == [error]
    assert len(state["variants"]) == 1


def test_a_run_that_never_passes_stops_at_the_revision_cap_and_ends_partial_and_flagged(
    brief: Brief,
) -> None:
    critique = ScriptedCritique([False])
    revise = FakeRevise()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=critique,
        revise=revise,
    )

    assert len(revise.calls) == 2
    assert len(critique.calls) == 3
    assert state["revision_count"] == 2
    assert [c.passed for c in state["critiques"]] == [False]
    assert state["status"] == "partial"
    assert state["result"] is not None
    assert state["result"].status == "partial"
    assert [item.flagged for item in state["result"].variants] == [True]
    assert state["result"].revision_count == 2


def test_a_passing_run_never_calls_the_reviser(brief: Brief) -> None:
    revise = FakeRevise()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=FakeCritique(),
        revise=revise,
    )

    assert revise.calls == []
    assert state["revision_count"] == 0


def test_a_failing_variant_is_revised_then_scored_again_until_it_passes(brief: Brief) -> None:
    critique = ScriptedCritique([False, True])
    revise = FakeRevise()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=critique,
        revise=revise,
    )

    assert len(revise.calls) == 1
    assert len(critique.calls) == 2
    assert state["revision_count"] == 1
    assert [v.headline for v in state["variants"]] == ["H-r1"]
    assert [c.passed for c in state["critiques"]] == [True]


def test_the_critic_scores_the_revised_variants(brief: Brief) -> None:
    critique = ScriptedCritique([False, True])

    run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=critique,
        revise=FakeRevise(),
    )

    assert critique.calls[0]["variants"][0].headline == "H"
    assert critique.calls[1]["variants"][0].headline == "H-r1"


def test_the_reviser_receives_the_failing_critiques(brief: Brief) -> None:
    revise = FakeRevise()

    run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=ScriptedCritique([False, True]),
        revise=revise,
    )

    seen = revise.calls[0]
    assert [c.passed for c in seen["critiques"]] == [False]
    assert seen["critiques"][0].fixes == ["Tighten the headline"]
    assert seen["revision_count"] == 0


def test_reviser_usage_is_added_to_the_run_total(brief: Brief) -> None:
    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=ScriptedCritique([False, True]),
        revise=FakeRevise(Usage(input_tokens=500, output_tokens=100, cost_usd=0.004)),
    )

    assert state["usage"].input_tokens == 610
    assert state["usage"].output_tokens == 155
    assert state["usage"].cost_usd == pytest.approx(0.007)


def test_the_assembler_runs_last_and_sees_the_final_state(brief: Brief) -> None:
    assemble = FakeAssemble()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=ScriptedCritique([False, True]),
        revise=FakeRevise(),
        assemble=assemble,
    )

    assert len(assemble.calls) == 1
    seen = assemble.calls[0]
    assert seen["revision_count"] == 1
    assert [c.passed for c in seen["critiques"]] == [True]
    assert [v.headline for v in seen["variants"]] == ["H-r1"]
    assert state["status"] == "failed"  # the fake assembler's update wins; nothing else sets it


def test_status_stays_running_until_the_assembler_sets_it(brief: Brief) -> None:
    seen: list[str] = []

    def spy(state: RunState) -> dict[str, Any]:
        seen.append(state["status"])
        return {}

    run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=FakeCritique(),
        assemble=spy,
    )

    assert seen == ["running"]


def test_usage_from_planner_writer_and_critic_is_summed(brief: Brief) -> None:
    critique = FakeCritique(Usage(input_tokens=1_000, output_tokens=200, cost_usd=0.01))

    state = run_graph(
        brief, load_brand("voltride"), plan=FakePlan(), write=FakeWrite(), critique=critique
    )

    assert state["usage"].input_tokens == 1_110
    assert state["usage"].output_tokens == 255
    assert state["usage"].cost_usd == pytest.approx(0.013)


def test_node_usage_is_added_to_existing_usage_by_the_reducer(brief: Brief) -> None:
    brand = load_brand("voltride")
    initial = new_run_state(brief, brand)
    initial["usage"] = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5)

    state = build_graph(FakePlan(), FakeWrite(), FakeCritique()).invoke(initial)

    assert state["usage"].input_tokens == 111
    assert state["usage"].output_tokens == 57
    assert state["usage"].cost_usd == pytest.approx(0.503)


def _structured_output_error(usage: Usage) -> StructuredOutputError:
    return StructuredOutputError(
        "reply was not valid", raw_text="{", usage=usage, outcome="complete"
    )


def test_a_planner_failure_ends_the_run_failed_without_calling_later_nodes(brief: Brief) -> None:
    write = FakeWrite()
    critique = FakeCritique()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(GatewayError("no plan")),
        write=write,
        critique=critique,
    )

    assert write.calls == []
    assert critique.calls == []
    assert state["status"] == "failed"
    assert state["variants"] == []
    assert [(e.node, e.message, e.fatal) for e in state["errors"]] == [
        ("planner", "GatewayError: no plan", True)
    ]
    assert state["result"] is not None
    assert state["result"].status == "failed"
    assert state["result"].variants == []
    assert state["result"].errors == state["errors"]


def test_a_writer_failure_ends_the_run_failed_and_keeps_the_plan_usage(brief: Brief) -> None:
    critique = FakeCritique()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(GatewayError("boom")),
        critique=critique,
    )

    assert critique.calls == []
    assert state["status"] == "failed"
    assert state["plan"] is not None
    assert [(e.node, e.fatal) for e in state["errors"]] == [("writer", True)]
    assert state["usage"].input_tokens == 10  # the planner's usage survives


def test_a_critic_failure_flags_the_unscored_variants_and_ends_partial(brief: Brief) -> None:
    revise = FakeRevise()

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=FakeCritique(error=GatewayError("judge down")),
        revise=revise,
    )

    assert revise.calls == []
    assert state["status"] == "partial"
    assert [e.node for e in state["errors"]] == ["critic"]
    assert state["result"] is not None
    assert state["result"].status == "partial"
    assert [item.flagged for item in state["result"].variants] == [True]
    assert [item.critique for item in state["result"].variants] == [None]


def test_a_reviser_failure_keeps_the_variants_and_flags_the_ones_still_failing(
    brief: Brief,
) -> None:
    critique = ScriptedCritique([False])

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=critique,
        revise=FakeRevise(error=GatewayError("reviser down")),
    )

    assert len(critique.calls) == 1  # no re-score after the failed revision
    assert state["status"] == "partial"
    assert state["revision_count"] == 0
    assert [v.headline for v in state["variants"]] == ["H"]
    assert [(e.node, e.fatal) for e in state["errors"]] == [("reviser", True)]
    assert state["result"] is not None
    assert [item.flagged for item in state["result"].variants] == [True]


def test_a_critic_failure_after_a_revision_flags_the_rewritten_variant(brief: Brief) -> None:
    class ScoresOnceThenFails(ScriptedCritique):
        def __call__(self, state: RunState) -> dict[str, Any]:
            if self.calls:
                self.calls.append(state)
                raise GatewayError("judge down")
            return super().__call__(state)

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=ScoresOnceThenFails([False]),
        revise=FakeRevise(),
    )

    assert state["status"] == "partial"
    assert state["revision_count"] == 1
    assert [v.headline for v in state["variants"]] == ["H-r1"]
    assert state["critiques"] == []  # the stale critique was dropped by the reviser
    assert state["result"] is not None
    assert [item.flagged for item in state["result"].variants] == [True]


def test_the_usage_of_a_failed_structured_call_is_added_to_the_run_total(brief: Brief) -> None:
    failed = Usage(input_tokens=700, output_tokens=300, cost_usd=0.02)

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(),
        critique=FakeCritique(error=_structured_output_error(failed)),
    )

    assert state["usage"].input_tokens == 810
    assert state["usage"].output_tokens == 355
    assert state["usage"].cost_usd == pytest.approx(0.023)
    assert "StructuredOutputError: reply was not valid" in state["errors"][0].message


def test_an_exception_with_no_message_is_still_recorded_with_its_type(brief: Brief) -> None:
    state = run_graph(
        brief, load_brand("voltride"), plan=FakePlan(RuntimeError()), write=FakeWrite()
    )

    assert [e.message for e in state["errors"]] == ["RuntimeError"]
    assert state["status"] == "failed"


def test_a_writer_warning_does_not_halt_the_run(brief: Brief) -> None:
    critique = FakeCritique()
    warning = RunError(node="writer", message="channel 'search': asked for 2 variants, got 1")

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(errors=[warning]),
        critique=critique,
    )

    assert len(critique.calls) == 1
    assert state["errors"] == [warning]
    assert warning.fatal is False


def test_a_failed_node_does_not_stop_the_other_errors_being_kept(brief: Brief) -> None:
    warning = RunError(node="writer", message="channel 'search': asked for 2 variants, got 1")

    state = run_graph(
        brief,
        load_brand("voltride"),
        plan=FakePlan(),
        write=FakeWrite(errors=[warning]),
        critique=FakeCritique(error=GatewayError("judge down")),
    )

    assert [(e.node, e.fatal) for e in state["errors"]] == [("writer", False), ("critic", True)]
    assert state["status"] == "partial"


def test_the_guard_does_not_swallow_keyboard_interrupts(brief: Brief) -> None:
    with pytest.raises(KeyboardInterrupt):
        run_graph(
            brief,
            load_brand("voltride"),
            plan=FakePlan(KeyboardInterrupt()),  # type: ignore[arg-type]
            write=FakeWrite(),
        )
