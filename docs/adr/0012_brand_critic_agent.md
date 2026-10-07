# 0012. Brand critic: strong-tier call per variant, verdict computed in code

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

The critic scores each variant against the brand's rubric, and its critiques drive the router
and the reviser (BF-16, BF-17). Two things can go wrong. A model that decides pass or fail
makes the quality gate drift with the prompt rather than with the configured thresholds. And
strict structured-output modes cannot express the free-form `Critique.scores` dict, so the
model cannot return a `Critique` directly (see the note in the architecture's state model).
The architecture also leaves open whether to score variants in parallel or in one call.

## Decision

`critique_variants(state)` is the graph node. It makes one `complete_structured` call on the
`strong` tier for each variant in `state["variants"]` and returns `critiques` (one per
variant, in variant order) and the summed `usage`.

- The model returns `CriticReply`: a list of `{criterion, score}` items and a list of fix
  notes. It is not asked for a variant id, an overall score or a verdict.
- `scoring.build_critique` converts the reply into a `Critique`. It requires every rubric
  criterion exactly once and every score in 1 to 5 (strict modes do not all support numeric
  bounds, so the range is checked here, as the planner does for its variant count). It
  computes `overall` as the mean and sets `passed` when `overall >= min_overall` and no
  criterion is below `min_per_criterion`, both from settings. A reply that does not fit the
  rubric raises `CritiqueError`; handling it is the graph's job (BF-22).
- The prompt is a versioned file, `prompts/critic_<version>.md`, chosen by the
  `critic_prompt_version` setting. It shows every anchor from 1 to 5 for each criterion, asks
  the critic to take the lower level when the copy sits between two, and keeps the variant and
  brief text in delimited sections marked as data, as in 0009 to 0011.
- The graph is now planner, writer, critic. The critic does not change the run status; the
  router and assembler own that (BF-16, BF-18).

## Alternatives considered

- **One call that scores all variants:** cheaper through shared context, but the model must
  echo variant ids back, and scoring side by side invites position and contrast effects that
  the eval's judge controls also avoid. Rejected for now. Per-variant calls can run in
  parallel if the latency measurement (day 3) shows they need to.
- **The model returns `passed` and `overall`:** fewer lines of code, but the gate would depend
  on the model's arithmetic and the prompt. Rejected; thresholds stay in config and the
  verdict is code.
- **Repairing a bad reply in code (dropping unknown criteria, clamping scores):** hides a
  critic that is not following the rubric. Rejected; the reply fails loudly and BF-21's
  re-ask is the place to retry it.

## Consequences

- **Gained:** a pass rule tunable from eval data without touching prompts, ids the model
  cannot get wrong, and critiques that are comparable across runs.
- **Trade-off accepted:** the brand profile and rubric are sent once per variant, so critic
  input tokens grow with the variant count, on the strong tier. After a revision the critic
  will score every variant again, including ones that already passed, unless BF-17 narrows
  it to the revised variants. A critique whose `passed` is false but whose `fixes` is
  empty gives the reviser nothing to act on; BF-17 must handle that case.
