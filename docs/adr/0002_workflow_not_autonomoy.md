# 0002. A fixed workflow with one evaluator loop, not autonomous agents

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

BrandForge turns a brief into scored, on-brand ad copy. The task has a known shape: plan,
retrieve examples, write, critique, and rewrite what fails. The goals ask for typed state,
bounded loops, graceful failure and full tracing, and every run must end in a defined state,
never a crash or an infinite loop. The question is how much control the models should have
over what runs next. A fully autonomous design lets an agent choose its own steps and tools.
A workflow fixes the steps in code and uses the model only inside each one.

## Decision

Build the pipeline as a workflow: a fixed LangGraph graph (planner, retriever, writer,
critic, router, reviser, assembler) with a single evaluator loop. The critic scores, the
router decides in code whether to revise, assemble or stop (0013), and the reviser rewrites
failing variants and returns to the critic (0014).

- Agents are pure functions over `RunState`. They never call each other, a provider or a
  tool. The graph owns control flow (0001) and every model call goes through the gateway
  (0008).
- Each node makes a text-in, structured-out call. No node plans its own steps, picks tools or
  decides when it is finished.
- The loop is the only cycle. Its exit conditions (all variants pass, revision cap, token
  budget) are plain code with unit tests.

## Alternatives considered

- **A supervisor agent that routes between worker agents:** more flexible for unusual
  briefs, but the number of calls, the cost and the path through the system would depend on
  model behaviour. That is harder to bound, test and trace, and harder to compare against the
  single-prompt baseline in the evals.
- **A single agent with tools and a free-form loop:** the simplest to write and the least
  predictable. Nothing in the pipeline needs tools, so the autonomy would buy nothing.
- **No loop at all (plan, write, critique, report):** cheapest and simplest, but the pipeline
  would only measure copy quality, never improve it. The critic loop is the main thing the
  evals test against the baseline.

## Consequences

- **Gained:** a predictable path and cost per run, nodes that are tested one at a time with a
  mocked gateway, a trace that reads the same way every run, and an eval that isolates what the
  pipeline adds over a single prompt.
- **Trade-off accepted:** less flexibility for unusual briefs. A brief that does not fit the
  plan, write, critique shape gets the same steps as any other, and supporting a new shape
  means changing the graph, not prompting the model differently.