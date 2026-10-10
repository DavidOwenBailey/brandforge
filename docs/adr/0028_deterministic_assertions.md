# 0028. Deterministic assertions score the eval row in promptfoo and pytest

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The eval asks whether the pipeline beats a single prompt on the 30 cases from
0026. Both systems already return one JSON row (0027). Each case records the
headline caps and the phrases the brief asked for. Banned words stay on the
brand profile, so a check that copied them onto the case would drift from the
profile the writer and the critic see. The baseline does not enforce length or
the requested channels (0009): an overrun has to show up in the score, not be
hidden by the generator. The same rules have to run inside promptfoo and inside
pytest, and neither path may call a model.

## Decision

`brandforge.evals.assertions` holds the checks. `evals/promptfooconfig.yaml`
puts them on `defaultTest`, so both providers are scored the same way.
`evals/providers/assertions.py` is the file promptfoo loads. It forwards to the
package, which is what the tests and mypy cover. A check returns pass or fail.
The score is 1 or 0.

- `is-json` checks that the output is JSON. `assert_json` checks that it is an
  `EvalOutput` for this case and this brand. A row can be JSON and still fail
  the schema check. `case_id` comes from the test vars, and the rendered prompt
  is the fallback, the same rule as the providers.
- Channel limits require every channel named on the brief, and no channel that
  is not. Each variant on a channel with `headline_max_chars` must have a
  headline of at most that many characters. `len` counts Unicode code points.
  The email subject is the headline. A channel the case does not cap, which in
  this set is social, is not length-checked. Every variant of a capped channel
  has to fit. The check does not require a variant count: the baseline count is
  a setting and the planner chooses its own.
- Banned words are a case-insensitive substring of the headline, the body, and
  the call to action, joined with newlines so two fields cannot form a word
  between them. The list is the brand profile's. This is the same match the
  example-corpus test uses.
- A call to action is present when every variant has one that is non-empty
  after stripping, and the row has at least one variant.
- `must_mention` uses the same substring rule. Every variant has to include
  every phrase. A case with no phrases passes that check once it has variants.
- A schema-valid row with no variants passes the JSON checks and fails the
  four content checks. `status` of `partial` or `failed` does not fail on its
  own. The copy is what the content checks score.
- A provider error row has no output. promptfoo fails that row before these
  checks. One bad case still does not stop the other provider.
- No new prompt file and no new setting.

## Alternatives considered

- **promptfoo `contains` and `not-contains` on each test.** Caps and banned
  words differ per case and per brand. The generator would grow a second copy
  of the rules, and pytest would not share it.
- **One global character cap per channel.** Rejected in 0026. The shipped
  briefs already use 50 and 60 for email subjects.
- **Copy banned words onto the case.** Rejected in 0026. The check would drift
  from the profile.
- **Reject a banned word only on a word boundary.** That would disagree with
  the corpus test. A substring also flags a longer word that merely contains
  the banned word, such as "detoxify" for "detox". The corpus rule wins.
- **Let one variant carry the required phrase.** The brief asked for the
  phrase, and each variant is a piece of copy a reader might see alone.
- **Require `baseline_variants_per_channel` copies of each channel.** The
  pipeline's count comes from the planner. The two systems would be scored
  against different numbers, and a shorter complete set would look like a miss.
- **Fail the JSON check unless `status` is `complete`.** A partial row is a
  real result. The copy checks already fail it when the copy is missing or
  over a cap.
- **Enforce the caps inside the baseline.** 0009 keeps the baseline from hiding
  an overrun. The eval is where it is counted.
- **Put the checks only inside `generate_tests`.** `defaultTest` keeps the case
  list free of check logic and applies one suite to both columns.

## Consequences

- **Gained:** `promptfoo eval` fails a row that is not the eval JSON, misses a
  requested channel, overruns a stated headline cap, uses a banned word, omits
  a call to action, or drops a required phrase. pytest covers the same
  functions with no model call.
- **Trade-off accepted:** a headline cap written only in prose is not checked.
  The case loader already rejects a cap that is not in both places.
- **Trade-off accepted:** body length is not checked. No case states a body cap.
- **Trade-off accepted:** the banned-word substring can flag a longer word,
  such as "detoxify" for "detox". Voice, clarity, and the strength of a call
  to action stay with the judge.
- **Trade-off accepted:** exclamation marks and the other prose "don't" rules
  are not deterministic. They are not a closed list on the profile.
