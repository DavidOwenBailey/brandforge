"""Gateway core tests. Providers are faked at the adapter boundary: no SDK, no network."""

import logging
import subprocess
import sys
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from brandforge.config import Budgets, Settings, Tier
from brandforge.llm.base import Outcome
from brandforge.llm.gateway import (
    GatewayConfigError,
    GatewayError,
    RawCompletion,
    StructuredOutputError,
    TransientProviderError,
    complete_structured,
)
from brandforge.models import Critique, Variant

VALID_VARIANT = (
    '{"id": "v1", "channel": "search", "headline": "Ride further", '
    '"body": "Built for the long way home.", "cta": "Shop now"}'
)


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


class FakeAdapter:
    def __init__(self, completion: RawCompletion) -> None:
        self._completion = completion
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "system": system,
                "schema": schema,
                "max_output_tokens": max_output_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._completion


def completion(
    text: str = VALID_VARIANT,
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    outcome: Outcome = "complete",
) -> RawCompletion:
    return RawCompletion(
        text=text, input_tokens=input_tokens, output_tokens=output_tokens, outcome=outcome
    )


class ScriptedAdapter:
    """Plays back one step per call: a completion is returned, an exception is raised."""

    def __init__(self, *steps: RawCompletion | Exception) -> None:
        self._steps = list(steps)
        self.call_count = 0

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
        step = self._steps[self.call_count]
        self.call_count += 1
        if isinstance(step, Exception):
            raise step
        return step


def retry_settings(**budgets: Any) -> Settings:
    return IsolatedSettings(budgets=Budgets(**budgets))


def test_returns_validated_model_and_usage() -> None:
    adapter = FakeAdapter(completion())

    variant, usage = complete_structured(
        "write copy", Variant, "strong", adapter=adapter, settings=IsolatedSettings()
    )

    assert variant == Variant(
        id="v1",
        channel="search",
        headline="Ride further",
        body="Built for the long way home.",
        cta="Shop now",
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    # strong tier: $2 in / $10 out per million tokens -> (100*2 + 50*10) / 1e6
    assert usage.cost_usd == pytest.approx(0.0007)


def test_request_passed_to_adapter() -> None:
    adapter = FakeAdapter(completion())
    settings = IsolatedSettings()

    complete_structured(
        "write copy", Variant, "strong", system="You write ads.", adapter=adapter, settings=settings
    )

    (call,) = adapter.calls
    assert call["model"] == "claude-sonnet-5-5"
    assert call["prompt"] == "write copy"
    assert call["system"] == "You write ads."
    assert call["schema"] is Variant
    assert call["max_output_tokens"] == settings.budgets.max_output_tokens_per_call
    assert call["timeout_seconds"] == settings.budgets.request_timeout_seconds


def test_system_prompt_defaults_to_none() -> None:
    adapter = FakeAdapter(completion())

    complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.calls[0]["system"] is None


@pytest.mark.parametrize(
    ("tier", "model", "expected_cost"),
    [
        ("strong", "claude-sonnet-5-5", (100 * 2 + 50 * 10) / 1e6),
        ("fast", "claude-haiku-4-5-20251001", (100 * 1 + 50 * 5) / 1e6),
        ("judge", "claude-opus-5-5", (100 * 4 + 50 * 20) / 1e6),
    ],
)
def test_tier_selects_model_and_price(tier: Tier, model: str, expected_cost: float) -> None:
    adapter = FakeAdapter(completion())

    _, usage = complete_structured("p", Variant, tier, adapter=adapter, settings=IsolatedSettings())

    assert adapter.calls[0]["model"] == model
    assert usage.cost_usd == pytest.approx(expected_cost)


def test_core_is_provider_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any provider's adapter works: the core only sees `provider:model` and RawCompletion."""
    monkeypatch.setenv("BRANDFORGE_MODELS__FAST", "gemini:some-gemini-model")
    monkeypatch.setenv("BRANDFORGE_PRICING__FAST__INPUT_PER_MTOK", "0.5")
    monkeypatch.setenv("BRANDFORGE_PRICING__FAST__OUTPUT_PER_MTOK", "2")
    adapter = FakeAdapter(completion())

    variant, usage = complete_structured(
        "p", Variant, "fast", adapter=adapter, settings=IsolatedSettings()
    )

    assert variant.id == "v1"
    assert adapter.calls[0]["model"] == "some-gemini-model"
    assert usage.cost_usd == pytest.approx((100 * 0.5 + 50 * 2) / 1e6)


def test_invalid_json_raises_with_raw_text_and_usage() -> None:
    adapter = FakeAdapter(completion("Sure! Here is your ad copy:"))

    with pytest.raises(StructuredOutputError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    err = info.value
    assert err.raw_text == "Sure! Here is your ad copy:"
    assert err.validation_error is not None
    assert err.outcome == "complete"
    assert err.usage.input_tokens == 100  # the failed call is still accounted for
    assert err.usage.cost_usd > 0


def test_schema_violation_raises() -> None:
    bad = '{"id": "v1", "channel": "billboard", "headline": "x", "body": "", "cta": "y"}'
    adapter = FakeAdapter(completion(bad))

    with pytest.raises(StructuredOutputError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert info.value.validation_error is not None
    assert "Variant" in str(info.value)


@pytest.mark.parametrize("outcome", ["truncated", "refused"])
def test_unusable_outcomes_raise(outcome: Outcome) -> None:
    adapter = FakeAdapter(completion('{"id": "v1", "chan', outcome=outcome))

    with pytest.raises(StructuredOutputError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert info.value.outcome == outcome
    assert info.value.validation_error is None
    assert info.value.usage.output_tokens == 50


def test_error_hierarchy() -> None:
    assert issubclass(StructuredOutputError, GatewayError)
    assert issubclass(GatewayConfigError, GatewayError)


def test_unsupported_schema_rejected_before_any_call() -> None:
    adapter = FakeAdapter(completion("{}"))

    with pytest.raises(GatewayConfigError, match="scores"):
        complete_structured("p", Critique, "strong", adapter=adapter, settings=IsolatedSettings())

    assert adapter.calls == []


def test_provider_without_adapter_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRANDFORGE_MODELS__FAST", "openai:some-model")

    with pytest.raises(GatewayConfigError, match=r"openai.*Available: anthropic, gemini"):
        complete_structured("p", Variant, "fast", settings=IsolatedSettings())


@pytest.mark.parametrize(
    ("provider", "key_variable"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")],
)
def test_missing_api_key_fails_fast(
    provider: str, key_variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRANDFORGE_MODELS__FAST", f"{provider}:some-model")
    monkeypatch.delenv(key_variable, raising=False)

    with pytest.raises(GatewayConfigError, match=key_variable):
        complete_structured("p", Variant, "fast", settings=IsolatedSettings())


def test_importing_the_gateway_does_not_import_any_provider_sdk() -> None:
    """The core must stay provider-neutral: SDKs load only inside their adapter."""
    code = (
        "import sys, brandforge.llm.gateway; "
        "print(sorted(m for m in ('anthropic', 'google.genai', 'openai') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


# --- retries (BF-20) ---------------------------------------------------------------


def test_transient_failures_are_retried_until_the_call_succeeds(sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(
        TransientProviderError("timed out", kind="timeout"),
        TransientProviderError("429", kind="rate_limit"),
        completion(),
    )

    variant, usage = complete_structured(
        "p", Variant, "fast", adapter=adapter, settings=IsolatedSettings()
    )

    assert variant.id == "v1"
    assert adapter.call_count == 3
    assert len(sleeps) == 2
    # Only the call that returned is accounted for: failed attempts produced no usage.
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50


def test_gives_up_after_max_attempts_and_raises_the_last_error(sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(
        TransientProviderError("first", kind="timeout"),
        TransientProviderError("second", kind="rate_limit"),
        TransientProviderError("third", kind="server"),
        completion(),  # never reached
    )

    with pytest.raises(TransientProviderError, match="third") as info:
        complete_structured(
            "p",
            Variant,
            "fast",
            adapter=adapter,
            settings=retry_settings(max_llm_retries=3),
        )

    assert info.value.kind == "server"
    assert adapter.call_count == 3  # 3 attempts in total, not 3 retries after the first
    assert len(sleeps) == 2  # no wait after the final attempt


def test_a_single_attempt_means_no_retry(sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(TransientProviderError("down", kind="server"), completion())

    with pytest.raises(TransientProviderError):
        complete_structured(
            "p", Variant, "fast", adapter=adapter, settings=retry_settings(max_llm_retries=1)
        )

    assert adapter.call_count == 1
    assert sleeps == []


def test_backoff_grows_exponentially_with_jitter(sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(
        *(TransientProviderError("busy", kind="rate_limit") for _ in range(3)), completion()
    )

    complete_structured(
        "p",
        Variant,
        "fast",
        adapter=adapter,
        settings=retry_settings(max_llm_retries=4, retry_initial_wait_seconds=1),
    )

    # initial * 2**(n-1) plus up to 1s of jitter: 1, 2, 4 seconds as the floor
    first, second, third = sleeps
    assert 1 <= first <= 2
    assert 2 <= second <= 3
    assert 4 <= third <= 5


def test_backoff_is_capped(sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(
        *(TransientProviderError("busy", kind="rate_limit") for _ in range(3)), completion()
    )

    complete_structured(
        "p",
        Variant,
        "fast",
        adapter=adapter,
        settings=retry_settings(max_llm_retries=4, retry_max_wait_seconds=1.5),
    )

    assert len(sleeps) == 3
    assert max(sleeps) <= 1.5


@pytest.mark.parametrize(
    "error",
    [
        GatewayConfigError("ANTHROPIC_API_KEY is not set."),
        RuntimeError("stands in for a provider SDK error such as a 400 or 401"),
    ],
)
def test_other_errors_are_not_retried(error: Exception, sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(error, completion())

    with pytest.raises(type(error)) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert info.value is error
    assert adapter.call_count == 1
    assert sleeps == []


@pytest.mark.parametrize("outcome", ["truncated", "refused"])
def test_unusable_replies_are_not_retried(outcome: Outcome, sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(completion('{"id": "v1", "chan', outcome=outcome), completion())

    with pytest.raises(StructuredOutputError):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 1
    assert sleeps == []


def test_invalid_replies_are_not_retried(sleeps: list[float]) -> None:
    # Re-asking after a validation failure is BF-21, not this retry policy.
    adapter = ScriptedAdapter(completion("not json"), completion())

    with pytest.raises(StructuredOutputError):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 1
    assert sleeps == []


def test_each_retry_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    adapter = ScriptedAdapter(
        TransientProviderError("429 from provider", kind="rate_limit"), completion()
    )

    with caplog.at_level(logging.WARNING, logger="brandforge.llm.gateway"):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "TransientProviderError" in record.getMessage()
    assert "429 from provider" in record.getMessage()


def test_transient_error_is_a_gateway_error() -> None:
    assert issubclass(TransientProviderError, GatewayError)
    assert TransientProviderError("x", kind="timeout").kind == "timeout"
