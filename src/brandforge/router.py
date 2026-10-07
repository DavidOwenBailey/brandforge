"""The router: decides what happens after the critic scores the variants (BF-16).

A pure function of the state, the settings and the time. It makes no model call and changes
nothing, so every path through the revision loop can be tested with hand-built state. The graph
(BF-17) turns its answer into the next node: `revise` goes to the reviser and back to the critic,
while `assemble` and `stop` both go to the assembler. They differ in meaning: `assemble` means
every variant passed, `stop` means some did not and the run must not try again, so the assembler
flags them and the run ends `partial` (BF-18).
"""

import logging
import time
from typing import Literal

from brandforge.budget import Clock, RunBudget
from brandforge.config import Settings, get_settings
from brandforge.models import RunState

logger = logging.getLogger(__name__)

RouteAction = Literal["assemble", "revise", "stop"]


def failing_variant_ids(state: RunState) -> list[str]:
    """Ids of the variants that have not passed, in variant order.

    A variant with no critique counts as failing: nothing has shown that it is on brand.
    Critiques for ids that are not in `variants` are ignored.
    """
    passed = {critique.variant_id for critique in state["critiques"] if critique.passed}
    return [variant.id for variant in state["variants"] if variant.id not in passed]


def _clock(now: float | None) -> Clock:
    """The real clock, or one frozen at `now` so a test can say what time it is."""
    if now is None:
        return time.time
    frozen = now
    return lambda: frozen


def route(
    state: RunState, *, settings: Settings | None = None, now: float | None = None
) -> RouteAction:
    """Decide the next step from the critiques, the revision count and the run budget.

    In order:

    1. Every variant passed: `assemble`. This wins over the limits, so a run that is already
       good is never reported as stopped.
    2. `revision_count` has reached `budgets.max_revisions`: `stop`.
    3. The run budget is used up (BF-23): `stop`. That is tokens used so far at or over
       `budgets.max_tokens_per_run`, or time since `started_at` at or over
       `budgets.max_wall_clock_seconds`. A revision means a rewrite and another full critic pass,
       so a run with no budget left does not start one. The reason is logged as a warning.
    4. Otherwise: `revise`.

    A run with no variants has nothing failing, so it routes to `assemble`; the assembler turns
    that into a `partial` or `failed` status from the errors in state.

    `now` is the time in seconds since the epoch, for tests; it defaults to the current time.
    The gateway enforces the same budget inside a node (`brandforge.budget`); the router is what
    stops a *new* revision from starting.
    """
    cfg = settings or get_settings()

    if not failing_variant_ids(state):
        return "assemble"
    if state["revision_count"] >= cfg.budgets.max_revisions:
        return "stop"
    budget = RunBudget.for_state(state, cfg.budgets, clock=_clock(now))
    reached = budget.exhausted()
    if reached is not None:
        logger.warning("Not revising the failing variants: %s.", budget.describe(reached))
        return "stop"
    return "revise"
