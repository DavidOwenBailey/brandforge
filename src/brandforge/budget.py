"""The per-run budget: how many tokens and how much wall-clock time one run may use (BF-23).

Two places enforce the same two limits, `budgets.max_tokens_per_run` and
`budgets.max_wall_clock_seconds`:

- The router (BF-16) asks `RunBudget.exhausted()` before starting another revision, so a run
  that has used up its budget stops revising and the assembler flags what is left.
- The gateway calls `RunBudget.check()` before every model call, so a node in the middle of a
  loop (a critic scoring ten variants, say) cannot carry on spending after the budget is gone.
  It raises `BudgetExceededError`; the graph's error edge (BF-22) turns that into a recorded
  error and sends the run to the assembler.

Both read the same definition of "used up", so they cannot disagree. A limit counts as reached
when usage is *at or over* it, the same rule the router has always used for tokens.

The budget is made from the run's state: the tokens earlier nodes already spent and the time the
run started, both of which are in `RunState`. The graph's guard makes one per node and installs
it with `active_budget`; the gateway finds it with `current_budget`, so no agent has to pass it
along. With no budget installed (the baseline generator, a one-off script) the gateway makes no
budget check at all.

Neither limit is exact. A call that starts under the limit can end over it, so a run can finish
over budget by up to one call's tokens, and over time by up to one request timeout.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from brandforge.config import Budgets
from brandforge.llm.base import BudgetExceededError, BudgetKind
from brandforge.models import RunState, Usage

Clock = Callable[[], float]  # seconds since the epoch, like `time.time`


@dataclass
class RunBudget:
    """Tokens and time left for a run, and what has been spent through this budget so far.

    `started_at` is when the run began (`RunState["started_at"]`) and `tokens_before` is what
    earlier nodes had already spent when this budget was made. `spent` grows as the gateway
    records each call, and covers this budget's lifetime only (one node), not the whole run.
    """

    max_tokens: int
    max_seconds: float
    started_at: float
    tokens_before: int = 0
    clock: Clock = time.time
    spent: Usage = field(default_factory=Usage)

    @classmethod
    def for_state(
        cls, state: RunState, budgets: Budgets, *, clock: Clock = time.time
    ) -> "RunBudget":
        """The budget a run has left at this point: the limits from config, the usage from state."""
        return cls(
            max_tokens=budgets.max_tokens_per_run,
            max_seconds=budgets.max_wall_clock_seconds,
            started_at=state["started_at"],
            tokens_before=state["usage"].total_tokens,
            clock=clock,
        )

    @property
    def tokens_used(self) -> int:
        """Input and output tokens used by the whole run so far."""
        return self.tokens_before + self.spent.total_tokens

    @property
    def elapsed_seconds(self) -> float:
        return self.clock() - self.started_at

    def exhausted(self) -> BudgetKind | None:
        """Which limit has been reached, or `None` if the run still has room. Tokens come first."""
        if self.tokens_used >= self.max_tokens:
            return "tokens"
        if self.elapsed_seconds >= self.max_seconds:
            return "time"
        return None

    def describe(self, kind: BudgetKind) -> str:
        """The reached limit as a short phrase, for logs and error messages."""
        if kind == "tokens":
            return f"token budget reached ({self.tokens_used:,} of {self.max_tokens:,} tokens used)"
        return (
            f"wall-clock budget reached ({self.elapsed_seconds:.1f}s of {self.max_seconds:g}s used)"
        )

    def check(self) -> None:
        """Raise `BudgetExceededError` if a limit has been reached. Call before spending."""
        kind = self.exhausted()
        if kind is not None:
            raise BudgetExceededError(
                f"Run stopped: {self.describe(kind)}.", kind=kind, usage=self.spent
            )

    def record(self, usage: Usage) -> None:
        """Add the usage of one finished model call, failed and repaired calls included."""
        self.spent = self.spent + usage


_current: ContextVar[RunBudget | None] = ContextVar("brandforge_run_budget", default=None)


def current_budget() -> RunBudget | None:
    """The budget installed by the enclosing `active_budget`, or `None` outside a run."""
    return _current.get()


@contextmanager
def active_budget(budget: RunBudget) -> Iterator[RunBudget]:
    """Make `budget` the one the gateway checks and records into, for the `with` block."""
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)
