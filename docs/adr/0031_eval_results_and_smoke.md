# 0031. Commit the eval summary, and smoke-test five cases in CI

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The eval suite can already run all 30 cases through the baseline and the
pipeline, score the hard checks, and grade each variant (0026–0029). The
architecture asks for two further things. The full run is manual, and its
summary is committed under `evals/results/`, so a reader can see the comparison
without paying for another run. CI is set up to run a 5-case smoke eval
on every push. That job is paused until the repository has an Anthropic
key, so today CI does not call a model. Calibration stays a manual
command (0030).

A promptfoo export is the wrong file to commit. It carries every variant, and
it is large. The architecture asks for the comparison: mean score per
criterion, pass rate, flagged-variant rate, tokens, cost and latency per
brief, baseline beside pipeline.

## Decision

`brandforge eval-summary` reads a promptfoo JSON export and writes
`evals/results/full.md` and `evals/results/full.json`. The markdown is the
table. The JSON is the same numbers, one row per case per system, including
which deterministic check failed. Variant text stays in the export. Write that
export outside `evals/results/` (`.promptfoo/` and `evals/output/` are
git-ignored).

- Each brief counts once. The judge mean is the unweighted mean of the row
  scores. A row score is already the mean of the per-criterion means.
- A finished judge grade counts, including a mean of 1. A grade that failed
  (bad JSON, no variants, a gateway error) does not count as 0.
- The pass rate uses the critic's thresholds (`min_overall`, default 4.0, and
  `min_per_criterion`, default 3), recorded in the summary. A graded brief
  passes when its mean is at least the overall floor and every criterion mean
  is at least the per-criterion floor. The rate is a report. It does not gate
  CI. The baseline has no critic, so this is the pass rate the two systems
  share.
- Flagged-variant rate counts pipeline flags only. Baseline rows stay "not
  scored" (0027).
- Tokens and cost come from the eval row, which is the system under test.
  Judge tokens stay off that total (0029). A cross-check mean, when the export
  has one, is reported on its own line.
- The model names, prompt versions, retrieval switch, run budgets and retry
  waits in the summary are the settings of the process that wrote it.
  Summarize in the same environment that ran the eval. The export does not
  record the tier models.

CI runs `evals/promptfooconfig.smoke.yaml`. The providers, the six hard
checks, the judge rubric and the concurrency of 1 are the same as the full
config. The test list is five case ids, also named `SMOKE_CASE_IDS`:

- `brightleaf_01_spring_blossom` (social, email, a headline cap)
- `brightleaf_02_starter_box` (search, display, a headline cap)
- `ledgerly_01_vat_reminders` (search, email, a headline cap)
- `voltride_01_commuter_ebike` (search, social, no headline cap)
- `voltride_04_test_ride_weekends` (email, social, search, a required phrase)

Together they cover the three brands, the four channels, a cap, a case with
no cap, and a phrase the copy must mention.

The smoke job is separate from `uv run poe check`. pytest still mocks the
model. The job uses the default Anthropic tiers, so it needs
`ANTHROPIC_API_KEY` as a repository secret. It does not build the example
index. Fork pull requests skip the job, because they do not receive secrets.
The job fails when a provider errors, a hard check fails, or the judge cannot
grade. A low judge score does not fail it. That last gate depends on the
judge assertion's weight staying above zero: promptfoo records a weight of 0
as a pass even when the grader refused (0029). The timeout is 45 minutes.
The raw smoke export is a CI artifact, not a committed file. CI does not
compare those scores with `evals/results/`: the smoke run and the committed
run can use different models, and the smoke run has no example index.

## Alternatives considered

- **Commit the promptfoo JSON.** A reader would see the copy, and the diff
  would be unreadable. The summary is the comparison the architecture asks
  to keep.
- **Gate CI on the judge mean, or on a match with the committed file.** The
  judge moves between runs, and CI's models are the defaults, which may
  differ from the machine that wrote the summary. The hard checks are the
  gate. The mean is the measurement.
- **Put the five ids only in a CLI filter.** The smoke set would live in the
  workflow script. A file next to the full config is reviewable, and a test
  checks that the file still matches `SMOKE_CASE_IDS` and the shared
  provider block.
- **Run the smoke eval inside pytest.** `poe check` would then need a key
  and would call a model. The definition of done keeps real calls out of
  the unit tests. The workflow job is the exception.
- **Skip the smoke job when the secret is missing.** A quiet skip hides a
  repository that never configured the key. A same-repo run with no key
  fails and says which secret to set. Forks are the case that skip, because
  GitHub withholds secrets from them.
- **Treat a baseline `flagged: null` as a pass rate of zero.** That would
  make the baseline look worse than an unscored system. The shared pass
  rate is the judge against the critic thresholds. The flag rate stays
  pipeline-only.

## Consequences

- **Gained:** `evals/results/full.md` is the baseline-versus-pipeline table
  for the manual run. CI calls the model on five cases and fails when those
  cases do not return a checkable row.
- **Trade-off accepted:** the summary's model names are only as good as the
  environment that wrote the file. Re-summarize after a run, on that machine.
- **Trade-off accepted:** the smoke eval can flake when a model drops a
  required phrase or a headline cap. The remedy is to rerun the job. The
  case list is small so that flake is visible.
- **Trade-off accepted:** CI does not retrieve approved examples, and it
  uses the default tiers. A green smoke run is not a reproduction of the
  committed scores.
- **Until the manual run:** `evals/results/` can be empty. `uv run poe check`
  does not create the summary and does not call a model. The results test
  checks `full.json` and `full.md` once those files are committed.
- **Paused:** the smoke job is `if: false` until the repository has
  `ANTHROPIC_API_KEY`. An empty secret failed the job. The config and the
  missing-key failure stay in the workflow. Remove `if: false` and restore
  the fork check to turn the job on.
