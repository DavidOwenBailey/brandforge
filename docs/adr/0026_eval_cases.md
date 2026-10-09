# 0026. Eval cases are self-contained YAML files

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

The eval asks whether the pipeline beats a single prompt on a fixed set: 3 brands
times 10 briefs, 30 cases. The CLI already ships 10 sample briefs spread across
the three brands, not 10 each. The architecture stores the 30 cases under
`evals/cases/` and says each one carries the brief, the brand id, and any hard
constraints. Those constraints have to be checkable later without a judge
(length, a required phrase) and they have to be the same instruction the model
was given. The sample briefs already disagree on email length: Brightleaf and
Voltride say 50 characters, Ledgerly says 60. Social has no cap.

## Decision

Each case is one YAML file in `evals/cases/`. The filename stem is the case id,
and the name starts with `<brand_id>_`. The file embeds a `Brief`, names the
brand, and may add `hard_constraints`:

- `headline_max_chars` caps the headline (the email subject is that headline)
  for the channels it names. A missing channel is not length-checked. Every cap
  is also written in the brief constraints as `<channel> ... max <n> characters`,
  with the same number. A cap for a channel the brief does not request is an
  error.
- `must_mention` lists phrases the copy has to include. Each phrase appears in
  the brief constraints, so the model was asked for it.
- Brand banned words are not copied onto the case. They stay on the brand
  profile, which already versions them.
- A non-empty call to action stays a property of `Variant`, not a per-case flag.

The ten CLI sample briefs are the first cases for their brands. The brief text
is copied, and a test fails if the two copies diverge. The other twenty briefs
exist only in the eval set. `load_cases` validates the directory. It does not
call a model.

## Alternatives considered

- **One file per brand, with a list of ten briefs.** That matches the example
  corpus, but a case is the unit a later promptfoo run will score, and one file
  per case is what `evals/cases/` describes.
- **Point at `src/brandforge/briefs/` instead of embedding the brief.** Twenty of
  the briefs are not CLI samples, and a case would break if a sample file moved.
  A self-contained file is what the architecture describes. The sync test covers
  the ten that are shared.
- **One global character cap per channel.** Shorter to check, and wrong for this
  set: the shipped briefs already use 50 and 60 for email subjects.
- **Copy banned words onto every case.** The deterministic check would drift
  from the profile the critic scores against.
- **Ship the cases inside the package.** The architecture keeps `evals/` at the
  repo root, beside the promptfoo config, calibration scores and results.

## Consequences

- **Gained:** 30 cases CI can count and validate with no model call. A later
  promptfoo provider can load the same files for the baseline and the pipeline.
- **Trade-off accepted:** the ten sample briefs are stored twice. Editing one
  means editing the other, and the test says so.
- **Trade-off accepted:** a headline cap written only in prose, or only in
  `headline_max_chars`, does not load. Authors write both, in that word order.
- **Trade-off accepted:** the loader looks for a checkout (`pyproject.toml` and
  `src/brandforge`) and otherwise uses `evals/cases` from the working directory.
  The dataset is not part of an installed wheel.
