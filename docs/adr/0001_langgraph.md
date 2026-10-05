# 0001. Use LangGraph for orchestration

- **Status:** Accepted
- **Date:** 2026-10-05

## Context

BrandForge is a multi-agent pipeline (planner, retriever, writer, critic,
router, reviser, assembler) with a bounded revision loop. Every run must end
in a defined state, never a crash or an infinite loop, and any run should be
inspectable and resumable. This needs explicit control over every transition
and a single typed state per run.

## Decision

Orchestrate the pipeline with LangGraph (1.2.x). Agents are pure functions
over a typed `RunState`; the graph owns control flow through conditional
edges, and state is checkpointed after each node using the SQLite
checkpointer.

## Alternatives considered

- **CrewAI:** role-based setup is quicker to write, but gives less control
  over state and transitions.
- **AutoGen:** conversation-centric model; harder to bound and test a fixed
  workflow.

## Consequences

- **Gained:** explicit state machine, conditional edges for the revision
  loop, checkpointing for inspect/resume, native Langfuse tracing.
- **Trade-off accepted:** more code than CrewAI's role-based setup.