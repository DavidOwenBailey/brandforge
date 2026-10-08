"""Structured JSON logging (BF-26).

Logs are one JSON object per line on stderr, so they never mix with the CLI's tables on stdout.
For the length of a run, `run_id` (and `trace_id`, when the run is traced) is bound into a
context variable. Every log emitted while the run is going — a node, the gateway, a retry —
carries those fields, which is what ties a line to the run ID the CLI prints and to the Langfuse
trace (ADR 0019). The reasoning is in ADR 0020.

Call sites keep using the standard library logger. structlog renders those records, and its own,
through one formatter. That leaves `tenacity`'s retry log and the existing tests alone.

Nothing here runs at import time beyond telling structlog how to speak to the standard library.
`configure_logging` is what actually installs the handler, and the CLI calls it on startup. With
no call, a log behaves as it did before this module existed.
"""

import logging
import re
import sys
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import Token
from typing import Any, TextIO, cast

import structlog

from brandforge.config import Settings, get_settings

__all__ = ["bind_run", "configure_logging", "get_logger"]

_HANDLER_NAME = "brandforge"

# A key whose name looks like a secret is redacted. The message text is not scanned: a brief
# can legitimately contain the word "secret", and the value we must not print is a field.
_SECRET_KEY = re.compile(r"(api[_-]?key|secret|password|authorization|credential)", re.IGNORECASE)

_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def _redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Replace the value of any secret-looking key. Nested structures are left alone."""
    for key in list(event_dict):
        if isinstance(key, str) and _SECRET_KEY.search(key):
            event_dict[key] = "***"
    return event_dict


def _renderer(log_format: str) -> structlog.typing.Processor:
    if log_format == "json":
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(colors=False)


def _configure_structlog() -> None:
    """Point structlog at the standard library. The handler, added later, does the rendering.

    Cached on first use: the processor chain does not depend on the log level or the format,
    so it never needs to change. `configure_logging` only swaps the handler.
    """
    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            _redact_secrets,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


_configure_structlog()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """A structlog logger. Its events go to the same place as `logging.getLogger`."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def configure_logging(settings: Settings | None = None, *, stream: TextIO | None = None) -> None:
    """Install (or replace) brandforge's log handler from `settings`.

    `stream` defaults to stderr. Tests pass their own. Calling this again replaces the previous
    brandforge handler and leaves every other handler, including pytest's, where it was.

    The level is set on the `brandforge` logger, not on the root, so turning on DEBUG here does
    not turn on DEBUG for httpx and the rest. Their warnings still come through, as JSON.
    """
    cfg = settings if settings is not None else get_settings()
    level = logging.getLevelNamesMapping()[cfg.log_level]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[*_SHARED_PROCESSORS, _redact_secrets],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _renderer(cfg.log_format),
        ],
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.name == _HANDLER_NAME:
            root.removeHandler(existing)
            existing.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.name = _HANDLER_NAME
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    logging.getLogger("brandforge").setLevel(level)


@contextmanager
def bind_run(run_id: str, *, trace_id: str | None = None) -> Iterator[None]:
    """Bind `run_id` (and `trace_id`, when there is one) onto every log inside the block.

    Restores whatever was bound before, so a nested run does not leak into the one around it
    and a finished run does not mark the next line. `trace_id` is omitted when it is `None`:
    a run that was not traced should not grow a `"trace_id": null` field.
    """
    fields: dict[str, str] = {"run_id": run_id}
    if trace_id is not None:
        fields["trace_id"] = trace_id
    tokens: Mapping[str, Token[Any]] = structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
