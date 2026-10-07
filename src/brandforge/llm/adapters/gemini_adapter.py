"""Gemini adapter: calls `generate_content` with a JSON-schema response config.

This is the only module in the gateway that imports the `google.genai` SDK.
"""

from functools import lru_cache
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from brandforge.config import Settings
from brandforge.llm.base import (
    GatewayConfigError,
    Outcome,
    RawCompletion,
    TransientProviderError,
)

# The JSON Schema keywords Gemini's `response_json_schema` supports. Anything else
# Pydantic emits (minLength, default, propertyNames, ...) is dropped here. That is
# safe: the gateway core still validates every reply with Pydantic.
_SUPPORTED_KEYWORDS = frozenset(
    {
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "type",
        "format",
        "title",
        "description",
        "enum",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "oneOf",
        "properties",
        "additionalProperties",
        "required",
    }
)
# Keywords whose value is a mapping of names to sub-schemas, or contains sub-schemas.
_NAMED_SCHEMAS = frozenset({"properties", "$defs"})
_SUB_SCHEMAS = frozenset({"items", "prefixItems", "anyOf", "oneOf", "additionalProperties"})

# Finish reasons that mean the model declined or was blocked, not that it ran out of room.
_REFUSED_FINISH_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.LANGUAGE,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.SPII,
    }
)


def _convert(node: Any) -> Any:
    if isinstance(node, list):
        return [_convert(item) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        # Gemini allows no siblings on a $ref except $-prefixed keywords.
        return {
            key: ({n: _convert(s) for n, s in value.items()} if key == "$defs" else value)
            for key, value in node.items()
            if key.startswith("$")
        }
    source = dict(node)
    if "const" in source:  # Pydantic emits `const` for a single-value Literal.
        source["enum"] = [source.pop("const")]
    converted: dict[str, Any] = {}
    for key, value in source.items():
        if key not in _SUPPORTED_KEYWORDS:
            continue
        if key in _NAMED_SCHEMAS:
            converted[key] = {name: _convert(sub) for name, sub in value.items()}
        elif key in _SUB_SCHEMAS:
            converted[key] = _convert(value)
        else:
            converted[key] = value
    return converted


def to_gemini_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Reduce `schema`'s JSON Schema to the subset Gemini's structured output supports."""
    converted: dict[str, Any] = _convert(schema.model_json_schema())
    return converted


class _ModelsAPI(Protocol):
    """The slice of `genai.Client().models` the adapter uses."""

    def generate_content(
        self, *, model: str, contents: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse: ...


class GeminiClient(Protocol):
    """Anything shaped like the Gemini client. Lets tests inject a fake."""

    @property
    def models(self) -> _ModelsAPI: ...


@lru_cache
def _gemini_client(api_key: str) -> genai.Client:
    # No retry options are set, and the SDK's default is "never retry". Retry policy
    # belongs in the gateway core, in one place.
    return genai.Client(api_key=api_key)


def _outcome_from(response: types.GenerateContentResponse) -> Outcome:
    if not response.candidates:
        blocked = response.prompt_feedback is not None and bool(
            response.prompt_feedback.block_reason
        )
        return "refused" if blocked else "complete"
    reason = response.candidates[0].finish_reason
    if reason == types.FinishReason.MAX_TOKENS:
        return "truncated"
    if reason in _REFUSED_FINISH_REASONS:
        return "refused"
    return "complete"


def _text_from(response: types.GenerateContentResponse) -> str:
    if not response.candidates:
        return ""
    content = response.candidates[0].content
    parts = (content.parts if content else None) or []
    # Thought summaries are not part of the answer.
    return "".join(part.text for part in parts if part.text and not part.thought)


def _transient_from(exc: Exception) -> TransientProviderError | None:
    """Classify an SDK error as retryable, or None if repeating the call would not help."""
    # The SDK lets httpx's transport errors through unwrapped. TimeoutException is a
    # TransportError, so it must be checked first.
    if isinstance(exc, httpx.TimeoutException):
        return TransientProviderError("Gemini request timed out.", kind="timeout")
    if isinstance(exc, httpx.TransportError):
        return TransientProviderError("Could not reach the Gemini API.", kind="connection")
    if isinstance(exc, errors.APIError):
        if exc.code == 429:
            return TransientProviderError("Gemini rate limit reached (429).", kind="rate_limit")
        if exc.code >= 500 or exc.code == 408:
            return TransientProviderError(f"Gemini server error ({exc.code}).", kind="server")
    return None


class GeminiAdapter:
    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiAdapter":
        api_key = settings.gemini_api_key.get_secret_value()
        if not api_key:
            raise GatewayConfigError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return cls(_gemini_client(api_key))

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
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=to_gemini_schema(schema),
            http_options=types.HttpOptions(timeout=max(1, round(timeout_seconds * 1000))),
        )
        try:
            response = self._client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except (errors.APIError, httpx.TransportError) as exc:
            transient = _transient_from(exc)
            if transient is None:
                raise
            raise transient from exc

        usage = response.usage_metadata
        # Thinking tokens are billed as output, so they count as output here.
        output_tokens = (
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) if usage else 0
        )
        return RawCompletion(
            text=_text_from(response),
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=output_tokens,
            outcome=_outcome_from(response),
        )
