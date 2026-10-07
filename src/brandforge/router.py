"""The router: decides what happens after the critic scores the variants (BF-16).

A pure function of the state and the settings. It makes no model call and changes nothing, so
every path through the revision loop can be tested with hand-built state. The graph (BF-17) turns
its answer into the next node: `revise` goes to the reviser and back to the critic, while
`assemble` and `stop` both go to the assembler. They differ in meaning: `assemble` means every
variant passed, `stop` means some did not and the run must not try again, so the assembler
flags them and the run ends `partial` (BF-18).
"""

from typing import Literal

from brandforge.config import Settings, get_settings
from brandforge.models import RunState

RouteAction = Literal["assemble", "revise", "stop"]


def failing_variant_ids(state: RunState) -> list[str]:
    """Ids of the variants that have not passed, in variant order.

    A variant with no critique counts as failing: nothing has shown that it is on brand.
    Critiques for ids that are not in `variants` are ignored.
    """
    passed = {critique.variant_id for critique in state["critiques"] if critique.passed}
    return [variant.id for variant in state["variants"] if variant.id not in passed]


def route(state: RunState, *, settings: Settings | None = None) -> RouteAction:
    """Decide the next step from the critiques, the revision count and the token budget.

    In order:

    1. Every variant passed: `assemble`. This wins over the limits, so a run that is already
       good is never reported as stopped.
    2. `revision_count` has reached `budgets.max_revisions`: `stop`.
    3. Tokens used so far have reached `budgets.max_tokens_per_run`: `stop`. A revision means
       a rewrite and another full critic pass, so a run with no budget left does not start one.
    4. Otherwise: `revise`.

    A run with no variants has nothing failing, so it routes to `assemble`; the assembler turns
    that into a `partial` or `failed` status from the errors in state.

    The wall-clock limit is not checked here: state holds no start time. BF-23 owns it.
    """
    cfg = settings or get_settings()

    if not failing_variant_ids(state):
        return "assemble"
    if state["revision_count"] >= cfg.budgets.max_revisions:
        return "stop"
    if state["usage"].total_tokens >= cfg.budgets.max_tokens_per_run:
        return "stop"
    return "revise"
