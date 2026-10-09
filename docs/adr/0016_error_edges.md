# 0016. Error edges: a failed node is recorded and the run goes to the assembler

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

The gateway retries transient failures and repairs invalid replies (BF-20, BF-21), but a node
can still fail after that: a provider that stays down, a refused or truncated reply, a critic
reply that does not fit the rubric, a missing API key. Until now that exception left the graph,
so the caller got a stack trace and no result, and the tokens already spent were lost. The
design requires every run to end in a defined state: "never a crash".

## Decision

The planner, writer, critic and reviser each run inside a guard in `graph.py`. If the node
raises an `Exception`:

- the guard records a `RunError(node, message, fatal=True)` and returns no other keys, so state
  is exactly what it was before the node ran;
- if the exception is a `StructuredOutputError`, its usage is added to the run total, because
  failed calls still cost money;
- the edge after the node sends the run to the assembler, which packages what exists.

The assembler decides the status as before (ADR 0015): `failed` if there are no variants,
`partial` if there are variants and any error or flag. The edges after the planner, writer and
reviser are conditional (continue, or assemble if a fatal error is recorded). The edge after
the critic goes to `stop` on a fatal error, otherwise to the router.

- **`RunError.fatal`** is new, default `False`. The writer already records non-fatal errors
  (a channel that came back short) and the run must carry on past them, so the edges need to
  tell a warning from a halt. It defaults to `False`, so existing errors are unchanged.
- **The message** is the exception type, then its text when it has any, so an empty message
  still says what went wrong.
- **Only `Exception`** is caught. `KeyboardInterrupt` and `SystemExit` still stop the run.
- **The assembler is not guarded.** It is plain code over state with nothing to retry; an
  exception there is a bug and should be loud.

What each failure leaves behind:

| Node fails | State kept | Final status |
| :---- | :---- | :---- |
| Planner | nothing but the error | `failed` |
| Writer | the plan | `failed` (it returns all variants or none) |
| Critic, first pass | the variants, unscored and so flagged | `partial` |
| Reviser | the variants and critiques from before it ran, so the failing ones stay flagged | `partial` |
| Critic, after a revision | the rewritten variants, flagged as unscored | `partial` |

## Alternatives considered

- **Catch in the CLI:** the API (BF-39) and the evals need the same behaviour, and the
  assembler's result is the right place for it.
- **Infer a halt from `errors` alone** (the last error was recorded by this node): the writer's
  non-fatal errors use the same node name, so a short channel would end the run. Rejected.
- **A separate `failed_node` key in `RunState`:** also works, but a flag on the error keeps one
  list as the record of what went wrong and adds nothing to the initial state.
- **Retry the failed node in the graph:** retries already live in the gateway, written once
  (ADR 0008). A second policy here would multiply attempts.
- **Keep a configuration error (missing API key) as an exception:** it would give the CLI a
  cleaner message, but a run that cannot start should still end in a defined state, and the
  error is listed in the summary.

## Consequences

- **Gained:** `run_graph` does not raise for a failure inside the four guarded nodes. Every
  such run returns a `RunResult` with an explicit status and the errors that explain it, and
  its cost includes the failed calls.
- **Trade-off accepted:** the writer and reviser are all or nothing. If the second channel
  fails, the first channel's variants are lost with the exception. Keeping partial progress
  would mean each agent returning what it had alongside the error. That is a follow-up if eval
  data shows it matters.
- **Trade-off accepted:** catching `Exception` turns a programming bug in a node into a
  recorded error as well. It is logged with the node name, and the unit tests call the agents
  directly, so bugs surface there.
- **Not covered here:** checkpointing (BF-24). The trace span for a failed node arrives with
  BF-25.
- **Extended in BF-23:** the guard also installs the run budget around each node, and keeps the
  usage of a `BudgetExceededError` as it does for a `StructuredOutputError` (0017).
- **Extended in BF-31:** the retriever is guarded the same way. An empty retrieval is not a
  failure and does not take the error edge (0024).
