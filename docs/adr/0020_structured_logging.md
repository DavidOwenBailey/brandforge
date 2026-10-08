# 0020. Structured JSON logs, with the run id bound for the whole run

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

The design asks for structured JSON logs that carry the same `run_id` as the Langfuse trace, so
a line can be tied back to the run that produced it (ADR 0019). Until now the code logged through
the standard library with no configuration: a warning was a sentence on stderr, with no run id,
no timestamp and no common shape. The useful warnings are already in place — a node that failed,
a retry, a schema repair, a budget that stopped a revision — and they should not have to be
rewritten, or pass `run_id` by hand, to become correlatable.

Two things the logs must not do: mix into the tables the CLI prints, and print a secret. API
keys live on settings and are easy to pass to a log by accident.

## Decision

Logging lives in `brandforge/logging.py` and is switched on by the CLI at startup.

- **One JSON object per line, on stderr.** structlog renders both its own loggers and the
  standard-library ones through a single formatter (`ProcessorFormatter`). The event, level,
  logger name and a UTC timestamp are always present. `BRANDFORGE_LOG_FORMAT=console` renders the
  same fields for a person; `BRANDFORGE_LOG_LEVEL` (default `INFO`, case-insensitive) is applied
  to the `brandforge` logger only, so DEBUG here does not turn on DEBUG for httpx. The CLI's
  tables stay on stdout.
- **`run_id` is bound, not passed.** `run_graph` opens `bind_run` around the whole invocation,
  including when the run raises. The id is stored in a context variable, and the formatter merges
  it into every record emitted inside the block: the guard, the router, the gateway's retry log.
  `trace_id` is bound the same way when the run has one, and omitted when it does not, so an
  untraced run does not grow a null field. The binding is restored afterwards, so one run cannot
  mark the next line.
- **A run logs its own edges.** `run_started` carries the brand id. `run_finished` carries the
  status, the tokens and the cost, or nulls when the run raised before the assembler. The other
  call sites are unchanged and keep the standard-library logger, which is what `tenacity`'s
  `before_sleep_log` requires.
- **A secret-looking key is redacted.** A field whose name matches `api_key`, `secret`,
  `password`, `authorization` or `credential` is logged as `***`. The message text is not
  scanned. Nested values are not walked.
- **Configuration is explicit.** Importing the module only tells structlog to speak to the
  standard library. With no call to `configure_logging`, logs behave as they did before: a
  library user who has not configured logging gets no new output. The CLI calls it from the
  top-level callback, so every command is covered.

## Alternatives considered

- **Rewriting every call site onto a structlog logger and passing `run_id` as an argument:**
  misses any log added later, and breaks `tenacity`, which needs a standard-library logger.
  Binding the id once covers both.
- **`logging.basicConfig` with a JSON formatter and a `LoggerAdapter`:** no new dependency, but
  the adapter has to be threaded through every module, which is the passing-by-hand problem
  again. structlog's context variables are the part we actually need, and the project is already
  happy to take a library for one job (tenacity, langfuse).
- **JSON on stdout, or replacing the CLI tables with JSON:** worse for the person running
  `brandforge generate`, and it would break every test that reads the tables. Logs and results
  are different streams.
- **Logging the brief and the model reply:** that is what the Langfuse trace is for. A log line
  stays a small event. Prompts in the logs would also be the thing most likely to contain
  something that should not be kept.

## Consequences

- **Gained:** any line emitted during a run can be joined to the printed run ID and, when
  tracing is on, to the Langfuse trace, without a call site knowing the id.
- **Gained:** the existing warnings keep their wording, so the tests that read them are
  unchanged. They gain a `run_id` only once logging is configured and a run is bound.
- **Trade-off accepted:** a third-party warning emitted during a run (a WARNING from httpx, say)
  is rendered as JSON too, and carries the `run_id`, because there is one handler on the root
  logger. INFO from those libraries stays off.
- **Trade-off accepted:** the redaction only sees top-level field names. A secret interpolated
  into the message text is not caught.
- **Trade-off accepted:** this adds the `structlog` package.
- **Not covered here:** shipping the logs anywhere. The per-node cost table is ADR 0021. The
  self-hosted Langfuse setup is ADR 0022.
