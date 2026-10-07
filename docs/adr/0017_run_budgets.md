# 0017. Run budgets: tokens and wall-clock time, enforced by the gateway and the router

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

Every run must be bounded in cost and in time (the design's "bounded everything"). Config has
had `budgets.max_tokens_per_run` (60,000) and `budgets.max_wall_clock_seconds` (90) since
BF-04, but only the router used one of them: it stops starting revisions once the tokens used so
far reach the limit (0013). Two gaps remained:

- Nothing watched the clock. State held no start time, and the router is a pure function.
- The router only acts between nodes. The critic makes one call per variant and the reviser one
  per failing variant, so a node already in progress could keep spending long after the budget
  was gone.

The design puts the budget check in the gateway, which every model call goes through, and has
the router stop revisions. The gateway does not know about runs: agents call
`complete_structured(prompt, schema, tier)` and nothing more.

## Decision

A `RunBudget` (`brandforge/budget.py`) holds the limits, when the run started, the tokens
earlier nodes had already used, and what has been spent through it since. Its one definition of
"used up" is `exhausted()`: tokens used at or over `max_tokens_per_run`, or seconds since
`started_at` at or over `max_wall_clock_seconds`, tokens reported first.

- **State** gets `started_at` (seconds since the epoch), set by `new_run_state`. It is the one
  fact about a run that a node cannot recompute, and it keeps both checks reading the same clock.
- **The gateway** calls `budget.check()` before every attempt (the first call, each retry after
  its backoff, each schema repair) and `budget.record(usage)` after every reply, before
  validating it, so unusable and repaired replies count. A used-up budget raises
  `BudgetExceededError` (a `GatewayError`, with `kind` of `tokens` or `time`) before the call is
  made. It is not retried or repaired.
- **The graph's guard** (0016) makes a `RunBudget` from the state each guarded node receives and
  installs it for the duration of the node, with a context variable. The gateway finds it there,
  so no agent signature changes. With no budget installed (the baseline generator, a script) the
  gateway checks nothing.
- **A node the budget stops** takes the existing error edge: a fatal `RunError`, then the
  assembler, which ends the run `partial` (variants exist) or `failed` (none do). The error
  carries the usage the node had spent before it was stopped, and the guard adds it to the run
  total, because those calls were paid for.
- **The router** now uses the same `RunBudget.exhausted()` after the revision-cap check, so it
  stops revising on tokens or time. It takes an optional `now` so it stays testable, and logs
  the reason as a warning. Stopping here is not an error: the variants are flagged and the run
  is `partial`, as with the revision cap.

What each limit does when it is hit:

| Hit | Where | Result |
| :---- | :---- | :---- |
| After the critic, with variants still failing | Router | No revision starts; variants flagged; `partial`; no error recorded |
| During a node (critic or reviser loop, a retry, a repair) | Gateway | `BudgetExceededError`; fatal `RunError` for that node; assembler; `partial` or `failed` |

## Alternatives considered

- **Pass a budget argument through every agent:** explicit, but it changes the signature and
  the test fake of the planner, writer, critic and reviser, and the gateway call in each.
  The context variable is set in one place, the guard, and read in one place, the gateway.
  It is the same ambient style as `get_settings()`, which the gateway already uses.
- **Keep one shared budget object for the whole run:** it would need to live outside state, so
  it would not survive the checkpointer (BF-24) and would not be visible to the router. Making
  a fresh budget from state at the start of each node needs only `usage` and `started_at`, which
  are already in state.
- **Check the budget in the agents' loops:** every agent would repeat the check, and a new
  agent could forget it. The gateway is the one door model calls go through (0008).
- **Have the router do all of it, and skip the gateway check:** the router cannot stop a node
  that is already running, so one pass over many variants could overspend freely.
- **A monotonic clock for the start time:** immune to the wall clock being adjusted, but a
  monotonic reading means nothing in another process, which is where a checkpointed run is
  resumed. `time.time()` is serialisable and good enough for a 90-second limit.
- **Shorten each call's timeout to the time left:** would make the time limit tighter, but a
  call started with a second left would time out for sure. The check before each attempt
  already stops anything new from starting.

## Consequences

- **Gained:** both limits are enforced inside a node as well as between nodes, from one
  definition, and every model call is covered without touching an agent. A node that is stopped
  no longer loses the cost of the calls it made first.
- **Trade-off accepted:** neither limit is exact. The check happens before a call, so a call
  that starts under the limit can end over it: a run can finish over the token budget by up to
  one call, and over the time limit by up to one request timeout (60 seconds by default) plus a
  backoff wait.
- **Trade-off accepted:** a node that is stopped part-way is all or nothing, as in 0016. If the
  reviser rewrites two variants and the budget ends before the third, the two rewrites are lost
  and the variants keep their earlier copy. Only the usage is kept.
- **Trade-off accepted:** the context variable is less visible than an argument. A node called
  by hand (as the agents' unit tests do) runs with no budget. Budget behaviour is tested at the
  gateway and the graph, where it is installed.
- **Trade-off accepted:** a resumed run (BF-24) keeps its original `started_at`, so a run
  resumed after the time limit has passed would stop revising at once. Resuming will need to
  decide whether to reset it.
- **Known gap, not changed here:** a node that raises `StructuredOutputError` after earlier
  successful calls still loses those earlier calls' usage (only the failed call's is kept, 0016).
  The budget now has the full figure, so using it in the guard would fix this. Left for a
  follow-up because it changes an existing behaviour and its tests.
- **Not covered here:** the status shows that variants were flagged but not that the budget
  caused it when the router stops a run. The reason is in the log; surfacing it in the result
  would need a non-fatal `RunError` from the router's side.
