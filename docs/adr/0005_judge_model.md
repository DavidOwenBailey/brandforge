# 0005. Judge the eval with a stronger model than the one that wrote the copy

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

The eval asks whether the pipeline writes more on-brand copy than a single prompt,
on the same briefs (0027). A grade from the model that wrote the copy favours its
own phrasing. The pipeline critic cannot supply the comparison either: it runs on
the strong tier, which is the planner's model, and the baseline has no critic
(0009, 0012). The architecture keeps a judge that is stronger than the generator,
and different from it.

## Decision

The judge is its own tier, `judge`, default `anthropic:claude-opus-5-5`, with its
own prices in config (0004). The pipeline never asks for it. The planner and the
critic use `strong`. The writer, the reviser and the baseline use `fast`.

- The judge scores each variant on its own, against that brand's anchors from 1 to
  5, and returns scores only. Pass and fail are computed in code. How the rubric
  is sent, how the row mean is stored, and why the assertion weight is small are
  in 0029. The prompt is a versioned file, `prompts/judge_<version>.md`.
- Length, banned words, the requested channels and a call to action are
  deterministic checks (0028). The judge is not grading a missing field or a long
  headline.
- A second model may cross-check the same rubric. That call borrows the judge tier
  for one request. It is not a fourth tier, and it stays off until
  `BRANDFORGE_JUDGE_CROSSCHECK_MODEL` is a `provider:model` string and both prices
  are set (0029).
- Fifteen hand-scored outputs measure agreement with the judge. Quadratic weighted
  kappa below 0.60 means tune the anchors. The command still exits 0 (0030). CI
  does not run it. The five-case smoke job, when it is on, calls the judge and
  fails when the judge cannot grade. A low score does not fail that job (0031).
- The summary's pass rate uses the critic thresholds (overall at least 4.0, no
  criterion below 3). It is a report. Which threshold should gate a row is still
  open.

## Alternatives considered

- **Grade with the critic's model.** Cheaper, and it reuses a call the pipeline
  already makes. The baseline would still need a separate grade, and the grader
  would share a model with the planner. Rejected as the default.
- **One model for writing and judging.** The smallest config. Self-preference
  would sit inside the number the eval exists to publish. Rejected. Config can
  still point `judge` at the same model as `fast` or `strong`. The default does
  not.
- **A fourth tier for the cross-check.** Agents would have to be told never to
  ask for it, and 0004's three tiers would grow. Borrowing the judge tier for
  that one call keeps the boundary. Rejected (0029).
- **Human scores on all 30 cases.** The comparison would change every time the
  prompts changed, and it would not be repeatable in CI. People score a fixed
  sample of 15. The judge scores the suite (0030).

## Consequences

- **Gained:** under the default tiers, baseline and pipeline are graded by a model
  that wrote neither. Swapping the judge is a config change. The pipeline's cost
  and latency are unchanged, because the graph never calls this tier.
- **Trade-off accepted:** a full eval spends a judge call per variant, on top of
  both systems. Those tokens are reported on the grader and are not added to the
  system under test (0029).
- **Trade-off accepted:** the judge can disagree with a person. Kappa is the
  signal to change the anchors. A low figure is a result, not a failed command.
- **Trade-off accepted:** pointing `judge` at the generator's model removes the
  bias control. The gateway does not reject that configuration.
