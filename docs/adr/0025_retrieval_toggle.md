# 0025. Retrieval toggle skips the search and still runs the node

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

The retriever always searches (0024). Evals need a second arm that runs the same pipeline
with no approved examples, while the index is still there. `retrieval_enabled` was already
in config, default on, and nothing read it. An empty index is a different case: that is a
warning that the search found nothing. Switching retrieval off is a choice.

## Decision

`retrieve_examples` reads `retrieval_enabled` after it has checked that a plan exists.

- **Off** logs `retrieval_disabled` at info, returns `examples: []`, and does not call the
  search. No Chroma client is opened, and no `RunError` is recorded. The writer then runs
  with no examples section, as it does for an empty search.
- **On** behaves exactly as 0024: one search per channel, a warning when the list is empty,
  and a raised error when the index is corrupt.
- **The node stays in the graph either way.** Checkpoints, traces and CLI step numbers do
  not change between the two arms. The comparison is about the examples, not about a
  different graph.
- **A missing plan still raises**, with the flag on or off. Both arms fail the same way
  when the planner produced nothing.
- **`brandforge index` ignores the flag** (0023). The off arm can be scored against an
  index that exists.

## Alternatives considered

- **Drop the retriever node from the graph when the flag is off.** One less span and no
  empty write. The two eval arms would then differ in checkpoint shape, trace shape and
  which step the writer is, so a score gap could be blamed on the graph rather than on the
  examples.
- **Leave `examples` untouched when the flag is off.** A state that already held examples
  would leak them into the off arm. The node always writes the list, and off writes `[]`.
- **Treat off as a warning, like a missing index.** The operator asked for no examples.
  A warning would make that arm look like a forgotten `brandforge index`.
- **Gate `brandforge index` on the same flag.** Building the index and using it are
  separate. The off arm is only a fair comparison when the index is present and unused.

## Consequences

- **Gained:** `BRANDFORGE_RETRIEVAL_ENABLED=false` runs the pipeline with no retrieved
  examples, and the log says the search was skipped. The default stays on.
- **Trade-off accepted:** the off arm still pays for a retriever span and a checkpoint.
  Both are local and carry no model cost. The node returns no `usage` key in either mode.
- **Trade-off accepted:** off still requires a plan, so the flag cannot paper over a
  planner that failed to write one.
