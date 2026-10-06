# 0009. Baseline generator: one structured call on the fast tier

- **Status:** Accepted
- **Date:** 2026-10-06

## Context

The evaluation asks whether the full pipeline beats a single prompt. For that answer
to mean anything, the baseline must be a fair opponent: the same model as the writer,
the same brand profile, and no extra scaffolding (no planner, critic, retrieval or
revision). It also has to fit the gateway rules from 0008: response schemas must be
portable across providers, and agents never see a model name.

## Decision

`generate_baseline(brief, brand)` makes one `complete_structured` call on the `fast`
tier (the writer's tier) and returns the variants with that call's `Usage`.

- The prompt is a versioned file, `prompts/baseline_<version>.md`, chosen by the
  `baseline_prompt_version` setting. Placeholders are `{{name}}` and are filled in a
  single pass, so brief text can never inject a placeholder.
- The model returns `BaselineReply`, a list of `VariantDraft` items without ids. Ids
  (`baseline-1`, `baseline-2`, ...) are assigned in code.
- The number of variants per channel is a setting (`baseline_variants_per_channel`,
  default 3), because the brief carries no variant count until the planner exists.
- Brief text sits inside a `<brief>` section that the prompt marks as data, not
  instructions, matching the security design for untrusted input.
- The prompt gives no per-channel length guidance. Length limits are checked by the
  deterministic eval assertions, so a baseline that overruns them is measured, not hidden.

## Alternatives considered

- **Reuse `Variant` as the response schema:** makes the model invent ids, and ids must
  be unique and stable. Rejected; code assigns them.
- **Strong tier for the baseline:** a stronger model would blur what the pipeline adds.
  Rejected; same tier as the writer.
- **Inline prompt string:** rejected by the definition of done; prompts are versioned
  files so eval results can be tied to an exact prompt.

## Consequences

- **Gained:** a fair, cheap, reproducible baseline; prompt changes are visible in review
  and selectable by config.
- **Trade-off accepted:** the baseline may overrun channel limits or the requested
  channel set, since neither is enforced here. Both are checked at eval time (BF-35).
