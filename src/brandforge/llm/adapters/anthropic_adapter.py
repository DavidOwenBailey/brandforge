"""Anthropic adapter: calls the Claude Messages API with a JSON-schema output config.

This is the only module in the gateway that imports the `anthropic` SDK.
"""

from collections.abc import Iterable
from functools import lru_cache
from typing import Protocol

import anthropic
from anthropic.types import Message, MessageParam, OutputConfigParam, TextBlockParam
from pydantic import BaseModel

from brandforge.config import Settings
from brandforge.llm.base import (
    GatewayConfigError,
    Outcome,
    RawCompletion,
    TransientProviderError,
)


class _MessagesAPI(Protocol):
    """The slice of `anthropic.Anthropic().messages` the adapter uses."""

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[MessageParam],
        system: str | Iterable[TextBlockParam] | anthropic.Omit,
        output_config: OutputConfigParam,
        timeout: float,
    ) -> Message: ...


class AnthropicClient(Protocol):
    """Anything shaped like the Anthropic client. Lets tests inject a fake."""

    @property
    def messages(self) -> _MessagesAPI: ...


@lru_cache
def _anthropic_client(api_key: str) -> anthropic.Anthropic:
    # max_retries=0: the SDK's own retries would hide failures from our budget and
    # tracing. Retry policy belongs in the gateway core, in one place.
    return anthropic.Anthropic(api_key=api_key, max_retries=0)


def _cached_system(system: str) -> list[TextBlockParam]:
    """The static prefix as one text block with a 5-minute cache breakpoint.

    The breakpoint sits on this block and not on the user message. The user message changes
    every call, and a breakpoint there would write a new cache entry each time and never read
    one (see ADR 0021). A prefix shorter than the model's minimum is billed as ordinary input;
    the API does not error.
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _outcome_from(stop_reason: str | None) -> Outcome:
    if stop_reason == "max_tokens":
        return "truncated"
    if stop_reason == "refusal":
        return "refused"
    return "complete"


def _transient_from(exc: anthropic.APIError) -> TransientProviderError | None:
    """Classify an SDK error as retryable, or None if repeating the call would not help."""
    # APITimeoutError subclasses APIConnectionError, so it must be checked first.
    if isinstance(exc, anthropic.APITimeoutError):
        return TransientProviderError("Anthropic request timed out.", kind="timeout")
    if isinstance(exc, anthropic.APIConnectionError):
        return TransientProviderError("Could not reach the Anthropic API.", kind="connection")
    if isinstance(exc, anthropic.RateLimitError):
        return TransientProviderError("Anthropic rate limit reached (429).", kind="rate_limit")
    if isinstance(exc, anthropic.APIStatusError) and (
        exc.status_code >= 500 or exc.status_code == 408
    ):
        return TransientProviderError(f"Anthropic server error ({exc.status_code}).", kind="server")
    return None


class AnthropicAdapter:
    def __init__(self, client: AnthropicClient) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnthropicAdapter":
        api_key = settings.anthropic_api_key.get_secret_value()
        if not api_key:
            raise GatewayConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return cls(_anthropic_client(api_key))

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        schema: type[BaseModel],
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> RawCompletion:
        try:
            json_schema = anthropic.transform_schema(schema)
        except ValueError as exc:
            raise GatewayConfigError(
                f"{schema.__name__} cannot be expressed as an Anthropic response schema: {exc}"
            ) from exc

        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=_cached_system(system) if system else anthropic.omit,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": json_schema}},
                timeout=timeout_seconds,
            )
        except anthropic.APIError as exc:
            transient = _transient_from(exc)
            if transient is None:
                raise
            raise transient from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        # Anthropic's input_tokens already exclude tokens written to or read from the cache.
        usage = response.usage
        return RawCompletion(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            outcome=_outcome_from(response.stop_reason),
        )
