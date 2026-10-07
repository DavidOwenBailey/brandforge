# 0004. Set model choice by tier in config, not by model name in code

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

Agents differ in how much judgement they need and how often they run. The planner and critic
decide things that steer or gate the whole run. The writer and reviser run most often and do
more routine work. The evals also need a judge that is stronger than, and different from, the
generator. Prices and models change, and the gateway must work across providers (0008), so
naming a model inside an agent would tie the code to one provider and make every cost
experiment a code change.

## Decision

Agents ask the gateway for a tier, never a model. Tiers are `strong`, `fast` and `judge`, and
each resolves in config (`ModelTiers`) to a `provider:model` string. Each tier has its own
price in `Pricing`, per million input and output tokens, which is changed together with the
model.

| Tier | Default model | Used by |
| :---- | :---- | :---- |
| `strong` | `anthropic:claude-sonnet-5-5` | planner, brand critic |
| `fast` | `anthropic:claude-haiku-4-5-20251001` | writer, reviser, baseline |
| `judge` | `anthropic:claude-opus-5-5` | eval judge only |

- The baseline uses the writer's tier so the comparison is fair (0009).
- The judge differs from the generator to reduce self-preference bias; the pipeline itself
  never uses it.
- Switching a tier to another provider (for example `gemini:<model-id>`) is an environment
  change plus its prices, with no agent code touched.

## Alternatives considered

- **One model for every agent:** simplest, but it either overspends on the high-volume writer
  and reviser calls or underspends on judgement-heavy planning and critique. Rejected.
- **Hard-coded model IDs per agent:** quick to write, but a model upgrade or a provider swap
  would edit several files, and agents would know about providers. Rejected.
- **Pick the tier at run time from the brief:** could save more, but it makes cost and quality
  depend on a decision that is hard to test and compare. Rejected for the POC.
- **A tier per agent instead of per role:** finer control, but more settings than the eval
  data can justify. The three tiers can be split later without changing the gateway interface.

## Consequences

- **Gained:** lower cost on the calls that run most, a one-line change to try a different model
  or provider, and eval results that can say which agents need the strong tier.
- **Trade-off accepted:** the fast tier may need more revision passes than a stronger model
  would, and a price left out of date in config makes reported costs wrong. Each tier's model
  and price must be changed together.