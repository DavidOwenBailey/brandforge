# 0006. Validate every boundary with Pydantic, and keep model schemas portable

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

A brief, a brand file, an eval case and a model reply are all untrusted input.
If a bad value reaches an agent, the failure shows up later as a crash or as copy
the router cannot score. The architecture says every value that crosses an agent
boundary is a Pydantic model, and a reply that does not parse is a retry, not a
crash. Strict structured-output modes also cannot express a free-form dict, so
the shape the model returns cannot always be the shape the rest of the program
stores.

## Decision

Contracts live in `brandforge.models` and share one base: unknown fields are
errors (`extra="forbid"`). Identifiers are stripped and must be non-empty.
Scores that the program stores are integers from 1 to 5. Brand YAML, briefs, the
example corpus, eval cases and calibration files use the same rule, each with
its own model. A bad file fails at load time and never reaches an agent.

`RunState` is a `TypedDict`, not a Pydantic model. LangGraph accumulates `usage`
and `errors` with reducers, and the checkpointer stores that dict (0018). The
values inside the dict are the Pydantic models. An agent returns only the keys
it owns.

What a model is asked to return is a second schema, and that schema has to be
portable. The gateway rejects a response model that contains a free-form object,
before any call (`brandforge.llm.schema`, 0008). The in-memory model may still
hold a dict. Fields the model must not invent are left off the response schema
and filled in code.

- `Critique.scores` is a dict. The model returns `CriticReply`: a list of
  `{criterion, score}` and a list of fixes. `build_critique` checks that every
  rubric criterion appears once, that each score is 1 to 5, and sets `overall`
  and `passed` from `BRANDFORGE_THRESHOLDS__MIN_OVERALL` and
  `BRANDFORGE_THRESHOLDS__MIN_PER_CRITERION`. The model does not decide the
  verdict (0012). The score on the wire is a plain int, because not every
  provider can express a numeric bound. A score outside 1 to 5 fails the reply.
  It is not clamped.
- The judge returns the same list of scores and no fixes (`JudgeReply`, 0029).
  `build_critique` checks that list too.
- The writer, reviser and baseline return a list of drafts. Channel and id are
  assigned in code. The planner returns audience, angle and a variant count.
  Channels are copied from the brief. The count is clamped to the configured
  maximum, because that bound is a budget the code owns (0010). That clamp is
  the exception. Other out-of-range model output fails validation.
- A reply that fails validation is re-asked once, with the errors in the prompt
  (`prompts/repair_<version>.md`). Unknown keys are not dropped, and a short
  reply is not padded. A truncated or refused reply is not repaired (0008).
- `brand.version` and `rubric.version` are copied onto `RunResult`, so a result
  names the profile that produced it.

## Alternatives considered

- **dataclasses plus a separate JSON Schema.** Two definitions of the same
  object, and the schema sent to the provider can drift from the parser.
  Rejected. Pydantic produces both.
- **Ask the model for `Critique`, including `passed` and `overall`.** Less
  conversion code, and the gate would follow the model's arithmetic instead of
  config. Rejected (0012).
- **Repair a bad critic reply in code,** by dropping unknown criteria or
  clamping scores. A critic that ignored the rubric would look like a normal
  grade. Rejected. The repair retry is the place to ask again.
- **Make `RunState` a Pydantic model.** Reducers and the SQLite checkpointer
  want a dict. Validation stays on the values, which is where a bad reply
  enters.
- **Allow free-form dicts in the response schema.** Strict modes reject them or
  return an object with no keys. The gateway refuses the schema instead, so the
  failure happens before a paid call.

## Consequences

- **Gained:** a bad brief, brand file or model reply fails where it was
  produced. The CLI, the API and the eval row share `RunResult`. Adding a field
  is a contract change.
- **Trade-off accepted:** the wire shape and the memory shape differ for the
  critic, the judge, the writer and the planner. The conversion is part of the
  contract. Unit tests cover it with a fake gateway.
- **Trade-off accepted:** a provider that cannot express `minLength` or a
  numeric bound still receives a schema. The gateway, or `build_critique`,
  checks the reply. The Gemini adapter drops keywords that provider rejects,
  because validation does not depend on the provider enforcing them (0008).
- **Trade-off accepted:** `RunState` is not validated as one object. A node that
  writes the wrong type fails when the next reader uses the field, or when the
  checkpointer rebuilds the model from its allowlist (0018).
