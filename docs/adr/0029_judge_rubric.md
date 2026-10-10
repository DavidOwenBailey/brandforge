# 0029. The judge rubric scores each variant on the brand's 1–5 anchors

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The eval already returns one JSON row per system (0027) and scores the hard
checks without a model (0028). Voice, clarity and the call to action are what
the rubric is for. The architecture asks for an `llm-rubric` on the judge tier,
anchored at 1–5, with each variant scored on its own so the judge has no
neighbouring copy to prefer. The judge tier is already a different, stronger
model than the generator (0004). An optional Gemini model may cross-check that
grade. The pipeline's critic cannot supply this score: the baseline has no
critic, and the critic runs on the strong tier.

## Decision

`brandforge.evals.judge` grades a row. `generate_tests` puts one `llm-rubric`
on each case, using that brand's anchors. `defaultTest` still holds the
deterministic checks. promptfoo runs the default checks and then this one.
`evals/providers/judge.py` is the file promptfoo loads. It forwards to the
package.

- The assertion value is the anchored rubric, so the eval record shows the
  scale. The model does not see promptfoo's default grader prompt.
  `rubricPrompt` is `{{output}}`, the row. The grader parses that row and calls
  the gateway with `prompts/judge_<version>.md` (`judge_prompt_version`,
  default `v1`). The prompt carries the same anchors. The brand block is the
  cached prefix. The variant under review is the user text.
- One gateway call per variant, on the judge tier. The call returns a list of
  `{criterion, score}` items. `build_critique` checks that every rubric
  criterion is present once and that each score is 1 to 5. The model does not
  decide pass or fail, and it is not asked for fixes.
- The row score is the mean of the per-criterion means, still on the 1–5 scale.
  The reason lists that mean, each criterion mean, and each variant's scores.
  promptfoo stores the same breakdown as grader metadata. The metric name is
  `judge`.
- A finished grade passes the assertion, including a mean of 1. The weight is
  0, so the 1–5 mean stays out of the 0–1 deterministic average and still shows
  up as its own metric. A row that is not the eval JSON, a row with no
  variants, a reply that misses the rubric, or a gateway error fails the
  assertion. One failed grade does not stop the other provider.
- The rubric scored is the one on disk for the case's brand. When the row's
  `rubric_version` differs, the reason says so.
- The judge's tokens and cost are reported on the grader response. They are
  not added to the row's `usage`, which stays the system under test.
- Cross-check is off unless `BRANDFORGE_JUDGE_CROSSCHECK_MODEL` is a
  `provider:model` string and both of its prices are above 0. Then
  `generate_tests` adds a second `llm-rubric` with metric `judge_crosscheck`.
  That call still goes through the gateway as the judge tier, with the
  cross-check model and prices copied onto the tier for that call. The Opus
  cache-read override is not copied. An unset cross-check cache multiplier
  uses the provider's own rate. The pipeline's three tiers are unchanged
  (0004).

## Alternatives considered

- **Let promptfoo call the judge with its built-in provider.** The grader
  would bypass the gateway, so retries, schema checks and the mocked unit-test
  path would be a second stack. The judge tier in config would not be the
  model that graded. Rejected.
- **Send every variant in one call.** Cheaper, and it gives the judge a set
  to rank. The architecture rules that out: variants are scored alone.
- **Fail the row when the mean is under the critic's threshold.** The
  model-graded pass rule in the architecture is to report the mean and the
  distribution. The threshold that should gate a row is still open. A low
  score stays visible, and it does not hide a deterministic failure or get
  hidden by one.
- **Add a fourth model tier for the cross-check.** Agents would have to be
  told never to ask for it, and 0004's three tiers would grow. Borrowing the
  judge tier for the second call keeps that boundary. Rejected as a new tier.
- **Reuse the critic prompt.** That prompt asks for fixes and is the voice of
  the pipeline critic. The eval judge only returns scores, from a different
  model. Rejected.
- **One static rubric string in `defaultTest`.** The anchors differ per brand
  and live on the profile. A static string would drift from the file the
  critic uses.

## Consequences

- **Gained:** `promptfoo eval` reports a 1–5 judge mean for the baseline and
  the pipeline, on the same anchors the critic uses, from the judge tier.
  pytest covers the grader with a fake gateway.
- **Trade-off accepted:** a full run grades every variant of both systems, so
  the judge is the expensive part of the eval. That is the cost of scoring
  variants alone.
- **Trade-off accepted:** the headline promptfoo score stays the deterministic
  average. Read the `judge` metric, and the `judge_crosscheck` metric when the
  second model is configured, for the 1–5 means.
- **Trade-off accepted:** a judge error fails the row even when the copy passed
  the deterministic checks. A missing grade should not look like a pass.
- **Trade-off accepted:** turning on the cross-check grades the row twice.
