# 0015. Assembler: plain code that flags unproven variants and owns the final status

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

Every run must end in a defined state with a result the caller can use. Until now the writer's
node set `status`, so a run whose variants never passed (`stop`) still ended `complete`, and
the loop ended at `END` with no packaged result: the CLI read variants and usage straight out
of state. ADRs 0013 and 0014 left flagging and the final status to this task.

## Decision

`assemble_result(state)` is the last graph node. Both `assemble` and `stop` from the router
lead to it, and it leads to the end. It makes no model call and reads the full state. It
returns two keys, `result` and `status`.

- **Flagged** means the variant has no passing critique, taken from the router's
  `failing_variant_ids`, so the router and the assembler cannot disagree. A variant that was
  never scored is flagged too, because nothing shows it is on brand.
- **Status:** `failed` if there are no variants, `partial` if any variant is flagged or any
  `RunError` was recorded, otherwise `complete`. The assembler is the only node that sets it,
  so it stays `running` until the run is packaged. The writer node no longer sets it.
- **`RunResult`** is a new contract: run id, status, brand id and version, rubric version,
  one `VariantResult` per variant (the variant, its latest critique or `None`, and `flagged`),
  revision count, errors and total usage. `RunState` gets a `result` key, empty until then.
- **CLI:** `generate` prints the variants, then a summary (status, brand and rubric versions,
  counts, a table with one column per scored criterion, the critic's remaining fixes for
  flagged variants, any errors) and the token cost. It exits 1 when the status is `failed`.

## Alternatives considered

- **Keep the status in the writer node and have the assembler only package:** two nodes
  would own one field, and the writer cannot know how the loop ended.
- **Have the assembler drop flagged variants:** hides copy that may still be usable. The
  design says partial results are returned, clearly labelled (ADR 0003).
- **Let the model decide the status:** the loop's exits are already code (ADR 0013); the status
  follows from the same facts.
- **Build the result inside the CLI:** the API (BF-39) and the evals need the same result,
  so it belongs in the graph.

## Consequences

- **Gained:** one place that decides the final status and what is flagged, a typed result for
  the CLI, the API and the evals, and a status that is honest when the loop gives up.
- **Trade-off accepted:** `partial` covers both "some variants are flagged" and "an error was
  recorded", so the result must be read for the reason: `flagged` on a variant, or `errors`.
  A variant that fails a re-score after revision is flagged even if it passed earlier, because
  only the latest critique counts (ADR 0014).
- **Not covered here:** the trace ID joins `RunResult` with Langfuse tracing (BF-25), cost
  per node with BF-27, and error edges that route a failed node here with what exists with
  BF-22. Errors raised by nodes still propagate until then.
