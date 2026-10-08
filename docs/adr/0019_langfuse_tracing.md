# 0019. Langfuse tracing: one trace per run, a span per node, a generation per model call

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

The design asks for every agent step to be traceable with its inputs, outputs, tokens and
latency, and for the trace ID to come back with the result so an output links straight to how it
was produced. Until now a run left a checkpoint trail (0018) and a log, but nothing that shows
where the time and the money went, or what a model was actually sent and replied.

The design sketches this as "Langfuse callbacks on every node". Two facts about the code shape
how it is done here. Every model call goes through the gateway (0008), which already knows the
model, the tokens and the cost. And a node that fails does not raise: the guard turns the
failure into a fatal `RunError` (0016), so anything that watches nodes for exceptions would
report a failed node as a success.

## Decision

Tracing lives in one module, `brandforge/tracing.py`, and is wired in at three points.

- **One trace per run.** `run_graph` opens a root span, `brandforge.run`, around the whole
  invocation. The trace ID is derived from the run's `run_id` (`Langfuse.create_trace_id` with
  the `run_id` as seed), so it is known before the run starts, is the same for the same `run_id`
  on any machine, and is stored in `RunState` (`trace_id`), in the checkpoints and in the
  `RunResult`. The CLI prints it after the run ID. The trace is tagged `brand:<id>` and carries
  the run ID and the brand and rubric versions as metadata.
- **One span per node.** Every node, the assembler included, is wrapped in `_traced` in
  `graph.py`, outside the guard. The span is named after the node and carries the node's prompt
  version (`planner_v1`, the same name as the prompt file), a small summary of the state it was
  given, and a summary of what it returned: counts, tokens and cost, not contents. Because the
  guard swallows exceptions, `_traced` marks the span as an error when the update holds a fatal
  `RunError`. A revision round is a second `critic` or `reviser` span, with `revision_count` in
  the metadata.
- **One generation per model call.** The gateway core opens a Langfuse generation around each
  call, with the model, provider, tier, output limit, prompt, reply, token counts and cost. The
  cost is the gateway's own (from the tier's configured price), not one Langfuse infers from the
  model name, so it matches the number the CLI prints. A schema repair (0016) is a second
  generation, and a call that fails or does not validate ends its generation as an error with the
  reply still attached. Retries of a transient failure happen inside one generation. Tracing is
  in the provider-neutral core, so no adapter knows about it.

Nesting needs no plumbing: spans use OpenTelemetry's context, which the Langfuse SDK is built
on, so a generation lands under the node that made it, including under LangGraph's own threads.

**Tracing never changes a run.** It is on only when `BRANDFORGE_TRACING_ENABLED` is true (the
default) and both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, so a fresh clone with
no keys behaves exactly as before and `trace_id` is `None`. Any error raised by Langfuse while a
span is opened, updated or closed is logged as a warning and swallowed. The same goes for the
client failing to start. A model call made outside a traced run (the baseline generator, a
script) is not traced, rather than becoming a trace of its own.

The run's trace is flushed when the run ends, because the SDK exports in the background and a CLI
process would otherwise exit first.

## Alternatives considered

- **Langfuse's LangChain/LangGraph `CallbackHandler`:** what the design sketched. It names spans
  after LangGraph's internals and, because the guard returns normally, it cannot see that a node
  failed: the span would look fine. It also knows nothing about the gateway's model calls, so the
  generations would still be hand-made. Wrapping the nodes ourselves is about twenty lines and
  gives control over names, content and error marking. Not tried here; revisit if hand-made spans
  become a burden.
- **Tracing inside each provider adapter:** every adapter would repeat it, and the cost and
  the retry loop, which are the interesting part of a call, live in the core.
- **Recording the trace in state only, and building it afterwards from the checkpoints:** no
  extra dependency, but no live view and no latency, since checkpoints have no timings.
- **A random trace ID kept in a context variable:** works, but the ID would have to be passed
  to the assembler some other way, and it could not be found again from a `run_id`.
- **Passing a tracer to every agent:** changes every agent signature and test fake. The node
  wrapper and the gateway cover every call from two places.

## Consequences

- **Gained:** any run can be opened in Langfuse from its trace ID: the node timeline, each
  model call's prompt, reply, tokens, cost and latency, and which node failed. Cost per node
  and per run comes from the same numbers the CLI reports. No agent changed.
- **Gained:** the trace ID is derived from the `run_id`, so `brandforge inspect <run_id>` and a
  trace can be matched without storing anything extra, and the checkpoints carry it too.
- **Trade-off accepted:** prompts and replies are sent to Langfuse unredacted. That is fine for
  fictional brands and a self-hostable backend, but a real deployment would need a masking or
  opt-out setting first.
- **Trade-off accepted:** the flush at the end of a run waits for the export. With Langfuse
  unreachable, a run that finishes normally takes a few seconds longer (about three when the
  connection is refused) and the exporter logs its own warnings. The run's result is unaffected.
- **Trade-off accepted:** this adds the `langfuse` package, which brings OpenTelemetry with it.
- **Trade-off accepted:** the Langfuse SDK keeps one client per public key for the life of the
  process, so tests give each test its own key.
- **Not covered here:** prompt caching and the per-node cost table in the CLI (BF-27) and the
  self-hosted Langfuse setup (BF-28). The structured JSON logs that carry the `run_id` are ADR
  0020. The trace ID will be returned by the API when it exists (BF-39).
