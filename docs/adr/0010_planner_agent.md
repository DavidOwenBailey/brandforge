# 0010. Planner: strong-tier call that decides audience, angle and variant count

- **Status:** Accepted
- **Date:** 2026-10-06

## Context

The writer needs more than the raw brief: a sharpened audience, one central angle to build
every variant on, and a variant count. Deciding these is judgement work, and it happens once
per run, so it is cheap to give it the strong tier. The plan also feeds the critic and the
reviser, so its shape must be fixed and validated rather than free text.

## Decision

`create_plan(brief, brand)` makes one `complete_structured` call on the `strong` tier and
returns a `Plan` with that call's `Usage`. `plan_brief(state)` is the graph node: it reads
`brief` and `brand` and returns only `plan` and `usage`.

- The model returns `PlanReply` (`audience`, `angle`, `variants_per_channel`). Channels are
  not asked for: code copies them from the brief, so a plan can never drop or invent one.
- `variants_per_channel` is clamped in code to 1 to `planner_max_variants_per_channel`
  (default 5). The schema carries no numeric bounds because strict structured-output modes
  do not all support them; the bound is also stated in the prompt.
- The prompt is a versioned file, `prompts/planner_<version>.md`, chosen by the
  `planner_prompt_version` setting. Brief text sits in a `<brief>` section the prompt marks
  as data, as in 0009.
- The planner runs before the baseline-wrapping node in the graph. The baseline ignores the
  plan until the writer replaces it (BF-14).

## Alternatives considered

- **Reuse `Plan` as the response schema:** lets the model choose channels and exceed the
  variant range, so a bad reply would fail validation or silently change scope. Rejected.
- **Fixed variant count from config only:** simpler, but the architecture gives the planner
  that decision, and eval data can show whether it helps. The config cap keeps cost bounded.
- **Fast tier for planning:** cheaper, but the plan steers every later step. Rejected until
  eval results say otherwise; tiers are config, so this is a one-line change.

## Consequences

- **Gained:** a validated, typed plan; the channel set is guaranteed; cost per run is bounded
  by the variant cap.
- **Trade-off accepted:** one extra strong-tier call per run, and a silent clamp when the
  model asks for a count outside the range.
