"""Anthropic adapter tests. The Anthropic client is faked: no network, no API key, no cost."""

from collections.abc import Callable
from typing import Any

import anthropic
import httpx2  # the HTTP library the Anthropic SDK is built on; its errors wrap its objects
import pytest
from anthropic.types import Message, MessageParam, OutputConfigParam
from pydantic_settings import SettingsConfigDict

from brandforge.config import Settings
from brandforge.llm.adapters.anthropic_adapter import AnthropicAdapter
from brandforge.llm.base import GatewayConfigError, TransientProviderError
from brandforge.llm.gateway import complete_structured
from brandforge.models import Variant

VALID_VARIANT = (
    '{"id": "v1", "channel": "search", "headline": "Ride further", '
    '"body": "Built for the long way home.", "cta": "Shop now"}'
)


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


class FakeMessages:
    """Plays back one outcome per call: a Message is returned, an exception is raised.

    The last outcome repeats if the adapter calls more often than outcomes were given.
    """

    def __init__(self, *outcomes: Message | Exception) -> None:
        self._outcomes = outcomes
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
        outcome = self._outcomes[min(len(self.calls), len(self._outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes: Message | Exception) -> None:
        self.messages = FakeMessages(*outcomes)


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


# --- error translation and retries (BF-20) -----------------------------------------

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error[E: anthropic.APIStatusError](cls: type[E], status: int) -> E:
    return cls("error", response=httpx2.Response(status, request=REQUEST), body=None)


@pytest.mark.parametrize(
    ("make_error", "kind"),
    [
        (lambda: anthropic.APITimeoutError(request=REQUEST), "timeout"),
        (lambda: anthropic.APIConnectionError(request=REQUEST), "connection"),
        (lambda: status_error(anthropic.RateLimitError, 429), "rate_limit"),
        (lambda: status_error(anthropic.InternalServerError, 500), "server"),
        (lambda: status_error(anthropic.InternalServerError, 529), "server"),  # overloaded
        (lambda: status_error(anthropic.APIStatusError, 408), "server"),
    ],
)
def test_transient_sdk_errors_become_transient_provider_errors(
    make_error: Callable[[], anthropic.APIError], kind: str
) -> None:
    error = make_error()
    adapter = AnthropicAdapter(FakeClient(error))

    with pytest.raises(TransientProviderError) as info:
        call(adapter)

    assert info.value.kind == kind
    assert info.value.__cause__ is error


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: status_error(anthropic.BadRequestError, 400),
        lambda: status_error(anthropic.AuthenticationError, 401),
        lambda: status_error(anthropic.PermissionDeniedError, 403),
        lambda: status_error(anthropic.NotFoundError, 404),
    ],
)
def test_other_sdk_errors_pass_through_unchanged(
    make_error: Callable[[], anthropic.APIError],
) -> None:
    error = make_error()

    with pytest.raises(anthropic.APIError) as info:
        call(AnthropicAdapter(FakeClient(error)))

    assert info.value is error


def test_gateway_retries_a_flaky_client_until_it_succeeds(sleeps: list[float]) -> None:
    client = FakeClient(
        anthropic.APITimeoutError(request=REQUEST),
        status_error(anthropic.RateLimitError, 429),
        text_message(VALID_VARIANT),
    )

    variant, usage = complete_structured(
        "write copy",
        Variant,
        "fast",
        adapter=AnthropicAdapter(client),
        settings=IsolatedSettings(),
    )

    assert variant.headline == "Ride further"
    assert len(client.messages.calls) == 3
    assert len(sleeps) == 2
    assert usage.input_tokens == 100  # only the successful call is accounted for


def test_gateway_does_not_retry_a_bad_request(sleeps: list[float]) -> None:
    client = FakeClient(status_error(anthropic.BadRequestError, 400), text_message(VALID_VARIANT))

    with pytest.raises(anthropic.BadRequestError):
        complete_structured(
            "write copy",
            Variant,
            "fast",
            adapter=AnthropicAdapter(client),
            settings=IsolatedSettings(),
        )

    assert len(client.messages.calls) == 1
    assert sleeps == []
