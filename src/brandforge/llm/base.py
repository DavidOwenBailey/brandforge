"""Provider-neutral types shared by the gateway core and every provider adapter.

Nothing in this module imports a provider SDK. An adapter translates between one
provider's API and these types; the gateway core only ever sees these types.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ValidationError

from brandforge.models import Usage

# How a provider call ended, normalised across providers.
Outcome = Literal["complete", "truncated", "refused"]


class GatewayError(Exception):
    """Base class for every error raised by the gateway."""


class GatewayConfigError(GatewayError):
    """The gateway cannot run with the current configuration (not retryable)."""


TransientKind = Literal["timeout", "connection", "rate_limit", "server"]


class TransientProviderError(GatewayError):
    """A provider call failed in a way that may succeed if simply repeated.

    Adapters raise this for timeouts, dropped connections, rate limits and 5xx
    responses, translating their own SDK's exceptions. It is the only error the
    gateway core retries. The original SDK exception is chained as `__cause__`.
    """

    def __init__(self, message: str, *, kind: TransientKind) -> None:
        super().__init__(message)
        self.kind = kind


class StructuredOutputError(GatewayError):
    """The model replied, but not with a valid instance of the requested schema.

    Carries what the repair retry (BF-21) needs: the raw text and the validation
    error. It also carries the usage, because the failed call still cost money.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        usage: Usage,
        outcome: Outcome,
        validation_error: ValidationError | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.usage = usage
        self.outcome = outcome
        self.validation_error = validation_error


@dataclass(frozen=True, slots=True)
class RawCompletion:
    """What an adapter returns: the reply text and normalised accounting, unvalidated."""

    text: str
    input_tokens: int
    output_tokens: int
    outcome: Outcome


class ProviderAdapter(Protocol):
    """Translates one provider's API to and from `RawCompletion`.

    An adapter builds the provider's request (including its structured-output
    setting), makes the call and normalises the result. It does not validate the
    reply, compute cost, retry or trace: the gateway core does that identically for
    every provider.

    The one thing an adapter does about failures is classify them: a timeout,
    dropped connection, rate limit or 5xx from its SDK is raised as
    `TransientProviderError`, so the core's retry policy never needs to know which
    SDK is underneath. Any other SDK error is left to propagate unchanged.
    """

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        schema: type[BaseModel],
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> RawCompletion: ...
