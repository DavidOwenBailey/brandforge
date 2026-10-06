"""Anthropic adapter: calls the Claude Messages API with a JSON-schema output config.

This is the only module in the gateway that imports the `anthropic` SDK.
"""

from functools import lru_cache
from typing import Protocol

import anthropic
from anthropic.types import Message, MessageParam, OutputConfigParam
from pydantic import BaseModel

from brandforge.config import Settings
from brandforge.llm.base import GatewayConfigError, Outcome, RawCompletion


class _MessagesAPI(Protocol):
    """The slice of `anthropic.Anthropic().messages` the adapter uses."""

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[MessageParam],
        system: str | anthropic.Omit,
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
    # tracing. Retry policy belongs in the gateway core (BF-20), in one place.
    return anthropic.Anthropic(api_key=api_key, max_retries=0)


def _outcome_from(stop_reason: str | None) -> Outcome:
    if stop_reason == "max_tokens":
        return "truncated"
    if stop_reason == "refusal":
        return "refused"
    return "complete"


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

        response = self._client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=system if system is not None else anthropic.omit,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": json_schema}},
            timeout=timeout_seconds,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return RawCompletion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            outcome=_outcome_from(response.stop_reason),
        )
