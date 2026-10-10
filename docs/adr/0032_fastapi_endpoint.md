# 0032. The HTTP API returns the run result and binds to loopback

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The CLI already runs one brief through the graph and prints the assembler's
result, including the Langfuse trace ID (0015, 0019). BF-39 asks for the same
thing over HTTP: `POST /generate` returns the result and the trace ID, and the
OpenAPI docs render. The architecture says this endpoint is local-only for the
proof of concept. Auth, rate limiting and tenant isolation are on the
production path, not this one.

## Decision

`POST /generate` accepts a brand id and a brief. The brief is the same `Brief`
contract as a brief YAML file. The response is the `RunResult` the assembler
already builds. `trace_id` on that model is the trace ID. It is null when
tracing is off, and the field is always present.

- **HTTP 200 means the pipeline finished.** That includes a run whose status is
  `partial` or `failed`. The status in the body is the run's status. The client
  receives the result and the trace ID together. A finished failed run is still
  HTTP 200.
- **404** when the brand id is unknown. **422** when the body is not a brand id
  plus a brief. **500** when the run does not finish: a gateway error that
  escaped the graph, a checkpoint-store error, or a graph that returned no
  result. Those responses have no `RunResult`. The detail for a caught failure
  is the same sentence the CLI prints: `generation failed: ...`.
- **The handler calls `run_graph` with the same checkpoint file as the CLI.**
  `brandforge inspect` can open a run that came in over HTTP. There is no
  second prompt and no second pipeline.
- **No authentication.** `brandforge serve` binds to `127.0.0.1` and port `8000`
  unless `BRANDFORGE_API_HOST`, `BRANDFORGE_API_PORT`, `--host` or `--port`
  says otherwise. Binding any other address is an explicit choice.
- **OpenAPI** is generated from these models. Interactive docs are at `/docs`.
  The schema is at `/openapi.json`.

JSON is the model's fields. `flagged_count` and `usage.total_tokens` stay
properties, as they are for the CLI. A client counts flagged variants from the
list, and tokens from the four token fields.

## Alternatives considered

- **Wrap the body as `{result, trace_id}`.** The trace ID would appear twice.
  `RunResult` already carries it, which is what the CLI prints.
- **Return HTTP 4xx or 5xx when the run status is `failed`.** Clients that read
  a body only on 200 would drop the result and the trace ID. The pipeline's
  contract is that a finished run returns a result (0016).
- **Skip the checkpointer so the API holds no local state.** `brandforge
  inspect` could not open the run, and the API would not match the CLI.
- **Add an API token for the proof of concept.** A token would suggest the port
  is safe to publish. It is not. Loopback is the control. Auth belongs with the
  production path the architecture already names.

## Consequences

- **Gained:** one result for the CLI, the API and the evals; docs that stay in
  step with the Pydantic contracts; a local server that does not call a model
  until a request arrives.
- **Trade-off accepted:** anyone who can reach the port can spend the model
  key. Do not publish the port. Runs share one SQLite checkpoint file. That
  fits a single user on one machine, which is the proof of concept (0007).
- **Not covered here:** the Streamlit page (BF-40) and Docker (BF-41). They can
  call this app. Cross-origin requests are refused, because CORS is not
  enabled.
