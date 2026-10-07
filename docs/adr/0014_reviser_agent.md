# 0014. Reviser: fast-tier rewrite of failing variants only, loop wired through the router

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

When the router says `revise`, something has to turn the critic's notes into better copy, and
the loop has to stay bounded. Three things can go wrong. Rewriting everything wastes tokens and
can make good variants worse. A model that returns ids or channels can drop or duplicate a
variant. And the critic can fail a variant without giving any fixes (ADR 0012 flagged this for
BF-17), which leaves a rewrite with nothing to act on.

## Decision

`revise_variants(state)` is the graph node. It reads `brief`, `brand`, `variants`, `critiques`
and `revision_count`, and picks what to rewrite with the router's `failing_variant_ids`, so the
reviser and the router can never disagree about what is failing.

- One `complete_structured` call on the `fast` tier for each failing variant. The model returns
  `ReviserReply` (headline, body, call to action). The id and channel come from state, so a
  rewrite replaces its variant in place and the order is unchanged.
- Passing variants are returned untouched.
- Each pass adds one to `revision_count`. That is what `route` compares with
  `budgets.max_revisions`, so the loop runs at most twice by default.
- The node returns `critiques` without the entries for rewritten variants. Those scores
  described the old copy. A variant with no critique counts as failing, so state stays honest
  if the critic fails before it re-scores.
- A failing critique with empty `fixes`, or a failing variant with no critique, still gets a
  rewrite. The prompt then says to raise the lowest-scoring criteria, or to improve the copy
  against the rubric, instead of citing notes that do not exist.
- If nothing is failing it makes no call and returns an empty update, without counting a
  revision. The router does not send a clean run here, so this is a guard.
- The prompt is a versioned file, `prompts/reviser_<version>.md`, chosen by the
  `reviser_prompt_version` setting. The variant, the critic's notes and the brief sit in
  delimited sections marked as data, as in 0009 to 0012.

The graph is now: planner, writer, critic, then a conditional edge on `router.route`.
`revise` goes to the reviser and back to the critic. `assemble` and `stop` both go to the end
until the assembler (BF-18) exists. The reviser does not change the run status.

## Alternatives considered

- **One call that rewrites all failing variants:** cheaper through shared context, but the
  model must echo ids back and rewrites would influence each other. Rejected, for the same
  reasons as the critic's per-variant calls.
- **Rewrite every variant each pass:** simpler, but it spends tokens on copy that passed and
  risks regressions. Rejected; "revise only what failed" is a stated design principle.
- **Skip variants with empty fixes:** the loop would then end with a failing variant that was
  never touched. Rejected; a fallback instruction costs one prompt line.
- **Use the strong tier:** better rewrites, but the reviser runs as often as the writer. The
  tier is config, so this can be tuned from eval data (ADR 0004).

## Consequences

- **Gained:** a bounded loop whose exit conditions all live in the router, ids that the model
  cannot get wrong, and rewrites aimed only at what the critic flagged.
- **Trade-off accepted:** the critic still re-scores every variant after a revision, including
  ones that already passed, on the strong tier. A passed variant can therefore fail on a later
  pass. Narrowing the critic to variants without a passing critique would save tokens and avoid
  that, but it changes the critic's contract, so it is left as a follow-up decision.
- **Not covered here:** failed reviser calls still raise (BF-22), and the final status and
  flagging of variants still failing at `stop` belong to the assembler (BF-18).
