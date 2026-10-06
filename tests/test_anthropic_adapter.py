"""Anthropic adapter tests. The Anthropic client is faked: no network, no API key, no cost."""

from typing import Any

import anthropic
import pytest
from anthropic.types import Message, MessageParam, OutputConfigParam
from pydantic_settings import SettingsConfigDict

from brandforge.config import Settings
from brandforge.llm.adapters.anthropic_adapter import AnthropicAdapter
from brandforge.llm.base import GatewayConfigError
from brandforge.models import Variant


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


class FakeMessages:
    def __init__(self, response: Message) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[MessageParam],
        system: str | anthropic.Omit,
        output_config: OutputConfigParam,
        timeout: float,
    ) -> Message:
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "system": system,
                "output_config": output_config,
                "timeout": timeout,
            }
        )
        return self._response


class FakeClient:
    def __init__(self, response: Message) -> None:
        self.messages = FakeMessages(response)


def make_message(
    content: list[dict[str, str]],
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    stop_reason: str = "end_turn",
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


def text_message(text: str, **kwargs: Any) -> Message:
    return make_message([{"type": "text", "text": text}], **kwargs)


def call(adapter: AnthropicAdapter, *, system: str | None = None) -> Any:
    return adapter.complete(
        model="claude-test-model",
        prompt="write copy",
        system=system,
        schema=Variant,
        max_output_tokens=777,
        timeout_seconds=12.5,
    )


def test_returns_text_tokens_and_complete_outcome() -> None:
    client = FakeClient(text_message("{}", input_tokens=11, output_tokens=7))

    result = call(AnthropicAdapter(client))

    assert result.text == "{}"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.outcome == "complete"


def test_request_shape() -> None:
    client = FakeClient(text_message("{}"))

    call(AnthropicAdapter(client), system="You write ads.")

    (sent,) = client.messages.calls
    assert sent["model"] == "claude-test-model"
    assert sent["max_tokens"] == 777
    assert sent["timeout"] == 12.5
    assert sent["system"] == "You write ads."
    assert sent["messages"] == [{"role": "user", "content": "write copy"}]
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["properties"]) == {"id", "channel", "headline", "body", "cta"}
    assert fmt["schema"]["additionalProperties"] is False


def test_system_prompt_omitted_when_not_given() -> None:
    client = FakeClient(text_message("{}"))

    call(AnthropicAdapter(client))

    assert isinstance(client.messages.calls[0]["system"], anthropic.Omit)


def test_text_blocks_are_joined_and_other_blocks_ignored() -> None:
    client = FakeClient(
        make_message(
            [
                {"type": "text", "text": '{"a": '},
                {"type": "text", "text": "1}"},
            ]
        )
    )

    assert call(AnthropicAdapter(client)).text == '{"a": 1}'


@pytest.mark.parametrize(
    ("stop_reason", "outcome"),
    [
        ("end_turn", "complete"),
        ("stop_sequence", "complete"),
        ("max_tokens", "truncated"),
        ("refusal", "refused"),
    ],
)
def test_stop_reason_is_normalised(stop_reason: str, outcome: str) -> None:
    client = FakeClient(text_message("{}", stop_reason=stop_reason))

    assert call(AnthropicAdapter(client)).outcome == outcome


def test_from_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(GatewayConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicAdapter.from_settings(IsolatedSettings())


def test_from_settings_builds_adapter_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")

    assert isinstance(AnthropicAdapter.from_settings(IsolatedSettings()), AnthropicAdapter)
