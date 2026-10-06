"""Gemini adapter tests. The Gemini client is faked: no network, no API key, no cost."""

from typing import Any, Literal

import pytest
from google.genai import types
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from brandforge.config import Settings
from brandforge.llm.adapters.gemini_adapter import GeminiAdapter, to_gemini_schema
from brandforge.llm.base import GatewayConfigError, Outcome
from brandforge.llm.gateway import complete_structured
from brandforge.models import Variant

VALID_VARIANT = (
    '{"id": "v1", "channel": "search", "headline": "Ride further", '
    '"body": "Built for the long way home.", "cta": "Shop now"}'
)


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


class FakeModels:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self, *, model: str, contents: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._response


class FakeClient:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.models = FakeModels(response)


def make_response(
    parts: list[types.Part] | None = None,
    *,
    finish_reason: types.FinishReason | None = types.FinishReason.STOP,
    prompt_tokens: int | None = 100,
    answer_tokens: int | None = 40,
    thought_tokens: int | None = 10,
    with_usage: bool = True,
) -> types.GenerateContentResponse:
    parts = parts if parts is not None else [types.Part(text="{}")]
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts), finish_reason=finish_reason
            )
        ],
        usage_metadata=(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_tokens,
                candidates_token_count=answer_tokens,
                thoughts_token_count=thought_tokens,
            )
            if with_usage
            else None
        ),
    )


def call(adapter: GeminiAdapter, *, system: str | None = None) -> Any:
    return adapter.complete(
        model="gemini-test-model",
        prompt="write copy",
        system=system,
        schema=Variant,
        max_output_tokens=777,
        timeout_seconds=12.5,
    )


def test_returns_text_and_counts_thinking_tokens_as_output() -> None:
    client = FakeClient(make_response([types.Part(text="{}")]))

    result = call(GeminiAdapter(client))

    assert result.text == "{}"
    assert result.input_tokens == 100
    assert result.output_tokens == 50  # 40 answer + 10 thinking
    assert result.outcome == "complete"


def test_request_shape() -> None:
    client = FakeClient(make_response())

    call(GeminiAdapter(client), system="You write ads.")

    (sent,) = client.models.calls
    config = sent["config"]
    assert sent["model"] == "gemini-test-model"
    assert sent["contents"] == "write copy"
    assert config.system_instruction == "You write ads."
    assert config.max_output_tokens == 777
    assert config.response_mime_type == "application/json"
    assert config.http_options.timeout == 12500  # the SDK takes milliseconds
    assert set(config.response_json_schema["properties"]) == {
        "id",
        "channel",
        "headline",
        "body",
        "cta",
    }


def test_system_instruction_is_none_when_not_given() -> None:
    client = FakeClient(make_response())

    call(GeminiAdapter(client))

    assert client.models.calls[0]["config"].system_instruction is None


def test_thought_parts_are_excluded_and_text_parts_joined() -> None:
    client = FakeClient(
        make_response(
            [
                types.Part(text="reasoning about the ad", thought=True),
                types.Part(text='{"a": '),
                types.Part(text="1}"),
            ]
        )
    )

    assert call(GeminiAdapter(client)).text == '{"a": 1}'


def test_missing_usage_metadata_counts_as_zero_tokens() -> None:
    client = FakeClient(make_response(with_usage=False))

    result = call(GeminiAdapter(client))

    assert (result.input_tokens, result.output_tokens) == (0, 0)


@pytest.mark.parametrize(
    ("finish_reason", "outcome"),
    [
        (types.FinishReason.STOP, "complete"),
        (types.FinishReason.OTHER, "complete"),
        (types.FinishReason.MAX_TOKENS, "truncated"),
        (types.FinishReason.SAFETY, "refused"),
        (types.FinishReason.RECITATION, "refused"),
        (types.FinishReason.BLOCKLIST, "refused"),
        (types.FinishReason.PROHIBITED_CONTENT, "refused"),
        (types.FinishReason.SPII, "refused"),
    ],
)
def test_finish_reason_is_normalised(finish_reason: types.FinishReason, outcome: Outcome) -> None:
    client = FakeClient(make_response(finish_reason=finish_reason))

    assert call(GeminiAdapter(client)).outcome == outcome


def test_blocked_prompt_with_no_candidates_is_refused() -> None:
    blocked = types.GenerateContentResponse(
        candidates=None,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        ),
    )

    result = call(GeminiAdapter(FakeClient(blocked)))

    assert result.outcome == "refused"
    assert result.text == ""


def test_no_candidates_and_no_block_reason_yields_empty_text() -> None:
    empty = types.GenerateContentResponse(candidates=None)

    result = call(GeminiAdapter(FakeClient(empty)))

    assert (result.outcome, result.text) == ("complete", "")


def test_from_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GatewayConfigError, match="GEMINI_API_KEY"):
        GeminiAdapter.from_settings(IsolatedSettings())


def test_from_settings_builds_adapter_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    assert isinstance(GeminiAdapter.from_settings(IsolatedSettings()), GeminiAdapter)


def test_gateway_runs_end_to_end_on_a_gemini_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRANDFORGE_MODELS__FAST", "gemini:gemini-test-model")
    monkeypatch.setenv("BRANDFORGE_PRICING__FAST__INPUT_PER_MTOK", "0.5")
    monkeypatch.setenv("BRANDFORGE_PRICING__FAST__OUTPUT_PER_MTOK", "2")
    client = FakeClient(make_response([types.Part(text=VALID_VARIANT)]))

    variant, usage = complete_structured(
        "write copy",
        Variant,
        "fast",
        adapter=GeminiAdapter(client),
        settings=IsolatedSettings(),
    )

    assert variant.headline == "Ride further"
    assert client.models.calls[0]["model"] == "gemini-test-model"
    assert usage.cost_usd == pytest.approx((100 * 0.5 + 50 * 2) / 1e6)


# --- schema conversion -------------------------------------------------------------

UNSUPPORTED = {"minLength", "maxLength", "pattern", "default", "propertyNames", "const"}


def keys_anywhere(node: Any) -> set[str]:
    """Every key used as a JSON Schema keyword (property names are skipped)."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            if key in ("properties", "$defs"):
                for sub in value.values():
                    found |= keys_anywhere(sub)
            else:
                found |= keys_anywhere(value)
    elif isinstance(node, list):
        for item in node:
            found |= keys_anywhere(item)
    return found


class _Inner(BaseModel):
    title: str  # a property that shares a name with a schema keyword
    kind: Literal["only"]
    tags: list[str] = Field(default_factory=list, max_length=3)


class _Outer(BaseModel):
    inner: _Inner = Field(description="described reference")
    maybe: str | None = None


def test_conversion_drops_unsupported_keywords_from_real_contracts() -> None:
    assert not keys_anywhere(to_gemini_schema(Variant)) & UNSUPPORTED


def test_conversion_keeps_supported_structure() -> None:
    schema = to_gemini_schema(Variant)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"id", "channel", "headline", "body", "cta"}
    assert schema["properties"]["channel"]["enum"] == ["search", "social", "display", "email"]


def test_conversion_handles_refs_defs_consts_and_keyword_named_properties() -> None:
    schema = to_gemini_schema(_Outer)

    # a $ref may not have siblings such as description
    assert schema["properties"]["inner"] == {"$ref": "#/$defs/_Inner"}
    inner = schema["$defs"]["_Inner"]
    assert "title" in inner["properties"]  # the property survives; it is not the keyword
    assert inner["properties"]["kind"]["enum"] == ["only"]  # const -> enum
    assert inner["properties"]["tags"]["maxItems"] == 3
    assert "default" not in inner["properties"]["tags"]
    assert not keys_anywhere(schema) & UNSUPPORTED


def test_conversion_keeps_nullable_unions() -> None:
    schema = to_gemini_schema(_Outer)

    assert {"type": "null"} in schema["properties"]["maybe"]["anyOf"]
