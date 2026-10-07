# 0011. Writer: fast-tier call per channel, with optional examples

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

The writer turns the plan into the variants the critic scores. It runs most often in the
pipeline, so it uses the cheap tier. The retriever that supplies approved examples arrives
much later (BF-31), so the writer must work well with no examples and take them without
changes when they appear. The pipeline is also compared with and without retrieval (open
question in the architecture), so "no examples" is a configuration to keep, not a gap.

## Decision

`write_variants(state)` is the graph node. It makes one `complete_structured` call on the
`fast` tier for each channel in the plan and returns `variants` and the summed `usage`.

- The model returns `WriterReply`, a list of `WriterDraft` items (headline, body, cta).
  Channel and id (`search-1`, `search-2`, ...) are assigned in code, so the model cannot
  add a channel or reuse an id.
- Drafts beyond `plan.variants_per_channel` are dropped. A channel that comes back short
  keeps what it got and adds a `RunError` for the writer, which makes the run `partial`.
- The prompt is a versioned file, `prompts/writer_<version>.md`, chosen by the
  `writer_prompt_version` setting. It carries the brand profile, each rubric criterion with
  its level-5 anchor (so the writer aims at what the critic rewards), the plan and the brief.
  Plan and brief text sit in delimited sections marked as data, as in 0009 and 0010.
- Examples are optional. The prompt gets an examples section only when `state["examples"]`
  holds examples for that channel; otherwise the section is absent, not empty.
- The graph is now planner then writer. The baseline wrapper is removed from the graph; the
  evals call `generate_baseline` directly.

## Alternatives considered

- **One call for all channels:** fewer calls, but the model controls the channel mix and the
  per-channel count. Rejected; code controls both, and per-channel calls can run in parallel
  later if wall-clock time needs it.
- **Failing the run on a short channel:** would throw away good variants. Rejected; keep them
  and record the shortfall.
- **Writer sees only the brand profile, not the rubric:** the critic would then judge against
  criteria the writer never saw, wasting revision passes. Rejected.

## Consequences

- **Gained:** guaranteed channels and ids, bounded output, a retrieval-off arm for the eval,
  and no writer change needed when the retriever lands.
- **Trade-off accepted:** the brand profile and rubric are sent once per channel, so input
  tokens grow with the channel count (at most four). Showing level-5 anchors may push the
  writer toward the rubric's wording, which the critic's calibration (BF-34) should check.
