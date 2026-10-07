# 0013. Router: a pure function that returns assemble, revise or stop

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

After the critic scores the variants, something must decide whether to rewrite the failing
ones, accept the result or give up. That decision is what keeps the revision loop bounded, so
it has to be predictable and cheap to test. It must not depend on a model, and it must hold
whatever the critic and reviser do.

## Decision

`router.route(state, settings=None)` returns one of `assemble`, `revise` or `stop`. It reads
`variants`, `critiques`, `revision_count` and `usage` from state, and the limits from
settings. It makes no model call and changes nothing. In order:

1. Every variant has a passing critique: `assemble`.
2. `revision_count >= budgets.max_revisions` (default 2): `stop`.
3. `usage.total_tokens >= budgets.max_tokens_per_run` (default 60,000): `stop`.
4. Otherwise: `revise`.

`failing_variant_ids(state)` is public. The reviser (BF-17) uses it to pick what to rewrite and
the assembler (BF-18) uses it to flag what is left. A variant with no critique counts as
failing, and a critique for an id that is not in `variants` is ignored.

`assemble` and `stop` both lead to the assembler. They stay separate so the graph, traces and
tests can tell a clean run from one that gave up. `stop` is what turns into flagged variants
and a `partial` status.

## Alternatives considered

- **Return a boolean "revise or not":** simpler, but the assembler would then have to work out
  again whether anything failed, and a trace could not show why the loop ended.
- **Return a richer decision with a reason:** useful for logging, but nothing consumes it yet.
  The action alone is enough until tracing (BF-25) shows a need.
- **Put the logic in the graph's conditional edge:** works, but buries the rule in wiring.
  A standalone function can be tested without compiling a graph.
- **Let the critic's `passed` be the only input and skip the limits:** the loop would then
  depend on the model eventually passing everything. Rejected; the limits come first.

## Consequences

- **Gained:** the loop's exit conditions live in one small function with tests for each,
  and both limits are tuned from config.
- **Trade-off accepted:** the budget check compares tokens already spent with the limit, not
  the cost of the next pass, so a run can finish slightly over budget. It also stops at
  "reached", not "exceeded", which gives up one revision a run that lands exactly on the
  limit could have had.
- **Not covered here:** the wall-clock limit. State has no start time, so BF-23 adds that check
  where the clock is available.
- **Wired in BF-17:** the graph calls `route` on a conditional edge after the critic, sending
  `revise` to the reviser (ADR 0014). `assemble` and `stop` both go to the end until the
  assembler (BF-18) exists.