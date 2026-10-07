# 0003. Cap the revision loop at two passes, and flag instead of failing

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

The critic loop is only safe if it ends. A model that is asked to fix copy may keep missing the
rubric, and each pass costs tokens and wall-clock time on both the fast tier (reviser) and the
strong tier (critic). The target is one brief end to end in about 60 seconds at a known cost.
When the loop gives up, the run must still produce something the caller can use.

## Decision

The loop runs at most `budgets.max_revisions` passes, default 2. When a variant still fails
after the cap, the router returns `stop` (0013) and the assembler marks that variant as
flagged instead of failing the run (0015).

- The cap is checked by the router against `revision_count`, which the reviser increments once
  per pass (0014). It is a limit in config, not a constant in code, so eval data can tune it.
- Two more limits sit beside it: `budgets.max_tokens_per_run` (default 60,000) and, since
  BF-23, `budgets.max_wall_clock_seconds` (default 90). Whichever is reached first ends the
  loop (0017).
- Flagged variants are returned with their latest critique and the remaining fixes, and the
  run status becomes `partial`. Passing variants in the same run are not held back.
- Only a run with no variants at all is `failed`.

## Alternatives considered

- **No cap, loop until everything passes:** best quality when it works, but cost and latency
  are unbounded, and a rubric the model cannot satisfy would loop forever. Rejected.
- **Fail the run when a variant is still below threshold:** simple and strict, but it throws
  away good variants and gives the caller nothing. Rejected; partial results are still useful.
- **Drop variants that never pass:** keeps the output clean, but hides copy that may be usable
  and removes the evidence of how often the pipeline gives up. Rejected; flag, do not hide.
- **A larger cap, such as 3 or 5:** more chances to pass, at a cost that grows with every pass
  and a latency target that gets harder to meet. Two is a starting point to revisit with eval
  data, in line with the open question on the pass threshold.

## Consequences

- **Gained:** bounded cost and latency per run, a loop whose exit conditions are testable code,
  and a result that is always labelled with what was and was not proven on brand.
- **Trade-off accepted:** some variants ship flagged instead of fixed. The `partial` status
  covers both flagged variants and recorded errors, so a caller must read the variant flags or
  the errors to know why (0015). The token check compares spend so far with the limit, so a run
  can finish slightly over budget (0013).