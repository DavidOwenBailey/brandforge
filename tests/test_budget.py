"""RunBudget (BF-23): when a run is out of tokens or time, and how the gateway finds the budget."""

import pytest
from pydantic_settings import SettingsConfigDict

from brandforge.brands import load_brand
from brandforge.budget import RunBudget, active_budget, current_budget
from brandforge.config import Budgets, Settings
from brandforge.llm.base import BudgetExceededError, GatewayError
from brandforge.models import Brief, Usage, new_run_state


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


class FakeClock:
    """A clock the test moves by hand: call it for the time, `advance` to move it on."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _budget(
    clock: FakeClock, *, max_tokens: int = 1_000, max_seconds: float = 90, tokens_before: int = 0
) -> RunBudget:
    return RunBudget(
        max_tokens=max_tokens,
        max_seconds=max_seconds,
        started_at=clock.now,
        tokens_before=tokens_before,
        clock=clock,
    )


def test_a_fresh_budget_has_room() -> None:
    clock = FakeClock()
    budget = _budget(clock)

    assert budget.exhausted() is None
    budget.check()  # does not raise


def test_tokens_are_exhausted_at_the_limit_not_just_over_it() -> None:
    clock = FakeClock()
    budget = _budget(clock, max_tokens=1_000)

    budget.record(Usage(input_tokens=600, output_tokens=399))
    assert budget.exhausted() is None

    budget.record(Usage(input_tokens=1))
    assert budget.tokens_used == 1_000
    assert budget.exhausted() == "tokens"


def test_tokens_already_spent_by_earlier_nodes_count_against_the_limit() -> None:
    clock = FakeClock()
    budget = _budget(clock, max_tokens=1_000, tokens_before=900)

    assert budget.exhausted() is None
    budget.record(Usage(input_tokens=60, output_tokens=40))
    assert budget.exhausted() == "tokens"


def test_time_is_exhausted_at_the_limit_not_just_over_it() -> None:
    clock = FakeClock()
    budget = _budget(clock, max_seconds=90)

    clock.advance(89.9)
    assert budget.exhausted() is None

    clock.advance(0.1)
    assert budget.elapsed_seconds == pytest.approx(90)
    assert budget.exhausted() == "time"


def test_time_runs_from_the_start_of_the_run_not_from_the_budget() -> None:
    # A node's budget is made part-way through a run: the clock still runs from `started_at`.
    clock = FakeClock(now=5_000.0)
    budget = RunBudget(max_tokens=1_000, max_seconds=90, started_at=4_920.0, clock=clock)

    assert budget.elapsed_seconds == pytest.approx(80)
    clock.advance(10)
    assert budget.exhausted() == "time"


def test_tokens_are_reported_first_when_both_limits_are_reached() -> None:
    clock = FakeClock()
    budget = _budget(clock, max_tokens=10, max_seconds=1)
    budget.record(Usage(input_tokens=10))
    clock.advance(5)

    assert budget.exhausted() == "tokens"


def test_check_raises_with_the_reason_and_what_was_spent() -> None:
    clock = FakeClock()
    budget = _budget(clock, max_tokens=100)
    spent = Usage(input_tokens=80, output_tokens=40, cost_usd=0.002)
    budget.record(spent)

    with pytest.raises(BudgetExceededError) as info:
        budget.check()

    assert info.value.kind == "tokens"
    assert info.value.usage == spent
    assert "token budget reached" in str(info.value)
    assert "120 of 100" in str(info.value)


def test_check_names_the_wall_clock_limit() -> None:
    clock = FakeClock()
    budget = _budget(clock, max_seconds=90)
    clock.advance(95)

    with pytest.raises(BudgetExceededError) as info:
        budget.check()

    assert info.value.kind == "time"
    assert "wall-clock budget reached" in str(info.value)
    assert "95.0s of 90s" in str(info.value)


def test_the_error_keeps_the_usage_as_it_was_when_it_was_raised() -> None:
    clock = FakeClock()
    budget = _budget(clock, max_tokens=10)
    budget.record(Usage(input_tokens=10))
    with pytest.raises(BudgetExceededError) as info:
        budget.check()

    budget.record(Usage(input_tokens=5))  # a later call must not rewrite the error's usage

    assert info.value.usage.input_tokens == 10


def test_a_budget_exceeded_error_is_a_gateway_error() -> None:
    assert issubclass(BudgetExceededError, GatewayError)


def test_record_accumulates_tokens_and_cost() -> None:
    budget = _budget(FakeClock())

    budget.record(Usage(input_tokens=10, output_tokens=5, cost_usd=0.001))
    budget.record(Usage(input_tokens=20, output_tokens=15, cost_usd=0.002))

    assert budget.spent.input_tokens == 30
    assert budget.spent.output_tokens == 20
    assert budget.spent.cost_usd == pytest.approx(0.003)


def test_for_state_reads_the_limits_from_config_and_the_rest_from_state() -> None:
    brief = Brief(
        product="e-bike", audience="commuters", objective="conversion", channels=["search"]
    )
    state = new_run_state(brief, load_brand("voltride"), started_at=123.0)
    state["usage"] = Usage(input_tokens=700, output_tokens=300)
    budgets = Budgets(max_tokens_per_run=5_000, max_wall_clock_seconds=45)
    clock = FakeClock(now=150.0)

    budget = RunBudget.for_state(state, budgets, clock=clock)

    assert budget.max_tokens == 5_000
    assert budget.max_seconds == 45
    assert budget.started_at == 123.0
    assert budget.tokens_used == 1_000
    assert budget.elapsed_seconds == pytest.approx(27)
    assert budget.spent == Usage()


def test_the_default_limits_are_the_documented_ones() -> None:
    brief = Brief(
        product="e-bike", audience="commuters", objective="conversion", channels=["search"]
    )
    state = new_run_state(brief, load_brand("voltride"))

    budget = RunBudget.for_state(state, IsolatedSettings().budgets)

    assert budget.max_tokens == 60_000
    assert budget.max_seconds == 90
    assert budget.exhausted() is None


# --- finding the budget from the gateway -------------------------------------------


def test_there_is_no_budget_outside_a_run() -> None:
    assert current_budget() is None


def test_active_budget_installs_the_budget_for_the_block_only() -> None:
    budget = _budget(FakeClock())

    with active_budget(budget) as installed:
        assert installed is budget
        assert current_budget() is budget

    assert current_budget() is None


def test_active_budgets_nest_and_restore_the_outer_one() -> None:
    outer, inner = _budget(FakeClock()), _budget(FakeClock())

    with active_budget(outer):
        with active_budget(inner):
            assert current_budget() is inner
        assert current_budget() is outer


def test_the_budget_is_uninstalled_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError), active_budget(_budget(FakeClock())):
        raise RuntimeError("node failed")

    assert current_budget() is None
