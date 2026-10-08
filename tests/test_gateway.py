"""Gateway core tests. Providers are faked at the adapter boundary: no SDK, no network."""

import logging
import subprocess
import sys
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from brandforge.budget import RunBudget, active_budget
from brandforge.config import Budgets, Settings, Tier
from brandforge.llm import gateway
from brandforge.llm.base import Outcome
from brandforge.llm.gateway import (
    BudgetExceededError,
    GatewayConfigError,
    GatewayError,
    RawCompletion,
    StructuredOutputError,
    TransientProviderError,
    complete_structured,
)
from brandforge.models import Critique, Usage, Variant

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
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    outcome: Outcome = "complete",
) -> RawCompletion:
    return RawCompletion(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        outcome=outcome,
    )


class ScriptedAdapter:
    """Plays back one step per call: a completion is returned, an exception is raised."""

    def __init__(self, *steps: RawCompletion | Exception) -> None:
        self._steps = list(steps)
        self.call_count = 0
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

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
        self.prompts.append(prompt)
        self.systems.append(system)
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


@pytest.mark.parametrize(
    ("provider_model", "tier", "expected_cost"),
    [
        # fast, Anthropic: input $1. (10 + 1000*1.25 + 4000*0.1 + 5*5) / 1e6
        ("anthropic:claude-haiku-4-5-20251001", "fast", (10 + 1_250 + 400 + 25) / 1e6),
        # fast, Gemini: no write premium. (10 + 1000*1 + 4000*0.1 + 25) / 1e6
        ("gemini:some-gemini-model", "fast", (10 + 1_000 + 400 + 25) / 1e6),
        # strong, Anthropic, read override 0.05 and input $2.
        # (10*2 + 1000*2*1.25 + 4000*2*0.05 + 5*10) / 1e6
        ("anthropic:claude-sonnet-5-5", "strong", (20 + 2_500 + 400 + 50) / 1e6),
        # strong pointed at Gemini keeps its 0.05 read override; the write rate is Gemini's 1x.
        # (10*2 + 1000*2*1 + 4000*2*0.05 + 5*10) / 1e6
        ("gemini:some-gemini-model", "strong", (20 + 2_000 + 400 + 50) / 1e6),
        # unknown provider: cache tokens billed as ordinary input, fast prices.
        ("other:some-model", "fast", (10 + 1_000 + 4_000 + 25) / 1e6),
    ],
)
def test_cache_tokens_are_priced_at_the_providers_rate(
    monkeypatch: pytest.MonkeyPatch, provider_model: str, tier: Tier, expected_cost: float
) -> None:
    """Cache writes and reads use the provider rate, unless the tier overrides one side.

    The fast tier has no override, so Anthropic's 1.25x write and 0.1x read apply, and
    Gemini's 1x write and 0.1x read apply. The strong tier's 0.05x read override wins even
    when that tier points at Gemini. An unknown provider is billed at the full input price.
    """
    monkeypatch.setenv(f"BRANDFORGE_MODELS__{tier.upper()}", provider_model)
    adapter = FakeAdapter(
        completion(
            input_tokens=10, output_tokens=5, cache_write_tokens=1_000, cache_read_tokens=4_000
        )
    )

    _, usage = complete_structured("p", Variant, tier, adapter=adapter, settings=IsolatedSettings())

    assert usage.cache_write_tokens == 1_000
    assert usage.cache_read_tokens == 4_000
    assert usage.total_tokens == 10 + 5 + 1_000 + 4_000
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
    assert len(adapter.calls) == 2  # the first reply and its one repair attempt
    assert err.usage.input_tokens == 200  # failed calls are still accounted for
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


def test_invalid_replies_are_not_backed_off(sleeps: list[float]) -> None:
    # A reply that does not validate is re-asked at once (BF-21), not waited on like a 429.
    adapter = ScriptedAdapter(completion("not json"), completion())

    complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

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


# --- schema repair (BF-21) ---------------------------------------------------------

BAD_CHANNEL = '{"id": "v1", "channel": "billboard", "headline": "x", "body": "b", "cta": "y"}'
FAST_CALL_COST = (100 * 1 + 50 * 5) / 1e6  # fast tier: $1 in / $5 out per million tokens


def test_invalid_reply_is_repaired_with_a_second_call() -> None:
    adapter = ScriptedAdapter(completion(BAD_CHANNEL), completion())

    variant, _ = complete_structured(
        "write the ad", Variant, "fast", adapter=adapter, settings=IsolatedSettings()
    )

    assert variant.channel == "search"
    assert adapter.call_count == 2


def test_repair_prompt_carries_the_request_the_bad_reply_and_the_problem() -> None:
    adapter = ScriptedAdapter(completion(BAD_CHANNEL), completion())

    complete_structured(
        "write the ad", Variant, "fast", adapter=adapter, settings=IsolatedSettings()
    )

    first, second = adapter.prompts
    assert first == "write the ad"  # the first call is untouched by the repair feature
    assert "write the ad" in second
    assert BAD_CHANNEL in second
    assert "- channel:" in second  # which field failed, as "<field>: <message>"


def test_a_repair_keeps_the_cached_system_prompt() -> None:
    adapter = ScriptedAdapter(completion(BAD_CHANNEL), completion())

    complete_structured(
        "write the ad",
        Variant,
        "fast",
        system="Brand rules stay here.",
        adapter=adapter,
        settings=IsolatedSettings(),
    )

    assert adapter.systems == ["Brand rules stay here.", "Brand rules stay here."]
    assert adapter.prompts[0] == "write the ad"
    assert "Brand rules stay here." not in adapter.prompts[1]


def test_repair_prompt_describes_an_unparseable_reply_as_a_whole() -> None:
    adapter = ScriptedAdapter(completion("Sure! Here is your ad copy:"), completion())

    complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert "Sure! Here is your ad copy:" in adapter.prompts[1]
    assert "- (whole reply):" in adapter.prompts[1]


def test_usage_covers_the_failed_call_and_the_repair() -> None:
    adapter = ScriptedAdapter(completion(BAD_CHANNEL), completion())

    _, usage = complete_structured(
        "p", Variant, "fast", adapter=adapter, settings=IsolatedSettings()
    )

    assert usage.input_tokens == 200
    assert usage.output_tokens == 100
    assert usage.cost_usd == pytest.approx(2 * FAST_CALL_COST)


def test_gives_up_when_the_repair_is_also_invalid() -> None:
    adapter = ScriptedAdapter(
        completion("not json"),
        completion(BAD_CHANNEL),
        completion(),  # last is never reached
    )

    with pytest.raises(StructuredOutputError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    err = info.value
    assert adapter.call_count == 2  # one repair, not a loop
    assert err.raw_text == BAD_CHANNEL  # the last reply, not the first
    assert err.validation_error is not None
    assert err.outcome == "complete"
    assert err.usage.input_tokens == 200
    assert err.usage.cost_usd == pytest.approx(2 * FAST_CALL_COST)


def test_the_number_of_repairs_is_configurable() -> None:
    adapter = ScriptedAdapter(
        completion("one"),
        completion("two"),
        completion(),
        completion(),  # last never reached
    )

    _, usage = complete_structured(
        "p", Variant, "fast", adapter=adapter, settings=retry_settings(max_schema_repairs=2)
    )

    assert adapter.call_count == 3
    assert usage.input_tokens == 300


def test_the_repair_retry_can_be_turned_off() -> None:
    adapter = ScriptedAdapter(completion("not json"), completion())

    with pytest.raises(StructuredOutputError) as info:
        complete_structured(
            "p", Variant, "fast", adapter=adapter, settings=retry_settings(max_schema_repairs=0)
        )

    assert adapter.call_count == 1
    assert info.value.usage.input_tokens == 100


def test_a_valid_first_reply_makes_exactly_one_call() -> None:
    adapter = ScriptedAdapter(completion(), completion())

    _, usage = complete_structured(
        "p", Variant, "fast", adapter=adapter, settings=IsolatedSettings()
    )

    assert adapter.call_count == 1
    assert usage.input_tokens == 100


def test_transient_failure_during_the_repair_is_retried(sleeps: list[float]) -> None:
    adapter = ScriptedAdapter(
        completion("not json"),
        TransientProviderError("429", kind="rate_limit"),
        completion(),
    )

    variant, usage = complete_structured(
        "p", Variant, "fast", adapter=adapter, settings=IsolatedSettings()
    )

    assert variant.id == "v1"
    assert adapter.call_count == 3
    assert len(sleeps) == 1
    # The failed transient attempt returned nothing to count: only the two replies are.
    assert usage.input_tokens == 200


def test_each_repair_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    adapter = ScriptedAdapter(completion(BAD_CHANNEL), completion())

    with caplog.at_level(logging.WARNING, logger="brandforge.llm.gateway"):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "Variant" in record.getMessage()
    assert BAD_CHANNEL not in record.getMessage()  # replies stay out of the logs


# --- run budget (BF-23) ------------------------------------------------------------

CALL_TOKENS = 150  # every `completion()` is 100 in + 50 out


class FakeClock:
    """A clock the test moves by hand: call it for the time, `advance` to move it on."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def run_budget(
    clock: FakeClock, *, max_tokens: int = 1_000, max_seconds: float = 90, tokens_before: int = 0
) -> RunBudget:
    return RunBudget(
        max_tokens=max_tokens,
        max_seconds=max_seconds,
        started_at=clock.now,
        tokens_before=tokens_before,
        clock=clock,
    )


def test_with_no_budget_installed_nothing_is_checked() -> None:
    # The baseline generator and one-off scripts have no run budget: no limit applies to them.
    adapter = ScriptedAdapter(
        completion(input_tokens=10**9), completion(input_tokens=10**9), completion()
    )

    for _ in range(3):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 3


def test_the_usage_of_every_call_is_recorded_into_the_run_budget() -> None:
    budget = run_budget(FakeClock())
    adapter = ScriptedAdapter(completion(), completion())

    with active_budget(budget):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert budget.spent.input_tokens == 200
    assert budget.spent.output_tokens == 100
    assert budget.spent.cost_usd == pytest.approx(2 * FAST_CALL_COST)
    assert budget.tokens_used == 2 * CALL_TOKENS


def test_an_invalid_reply_and_its_repair_are_both_recorded() -> None:
    budget = run_budget(FakeClock())
    adapter = ScriptedAdapter(completion(BAD_CHANNEL), completion())

    with active_budget(budget):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert budget.spent.input_tokens == 200


def test_an_unusable_reply_is_still_recorded() -> None:
    budget = run_budget(FakeClock())
    adapter = ScriptedAdapter(completion('{"id": "v1", "chan', outcome="truncated"))

    with active_budget(budget), pytest.raises(StructuredOutputError):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert budget.spent.output_tokens == 50


def test_an_exhausted_token_budget_stops_the_call_before_it_is_made(sleeps: list[float]) -> None:
    budget = run_budget(FakeClock(), max_tokens=1_000, tokens_before=1_000)
    adapter = ScriptedAdapter(completion())

    with active_budget(budget), pytest.raises(BudgetExceededError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 0
    assert info.value.kind == "tokens"
    assert info.value.usage == Usage()  # this node had spent nothing yet
    assert sleeps == []  # not retried


def test_the_budget_can_run_out_between_two_calls_of_one_node() -> None:
    # Each call is 150 tokens; the limit is 250. The second call starts under it and ends over it.
    budget = run_budget(FakeClock(), max_tokens=250)
    adapter = ScriptedAdapter(completion(), completion(), completion())

    with active_budget(budget):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())
        with pytest.raises(BudgetExceededError) as info:
            complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 2
    assert info.value.kind == "tokens"
    assert info.value.usage.input_tokens == 200  # what the two calls that were made cost
    assert info.value.usage.output_tokens == 100


def test_tokens_spent_by_earlier_nodes_count_before_the_first_call() -> None:
    budget = run_budget(FakeClock(), max_tokens=1_000, tokens_before=900)
    adapter = ScriptedAdapter(completion(), completion())

    with active_budget(budget):
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())
        with pytest.raises(BudgetExceededError):
            complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 1


def test_no_repair_is_made_once_the_invalid_reply_used_up_the_budget() -> None:
    budget = run_budget(FakeClock(), max_tokens=CALL_TOKENS)
    adapter = ScriptedAdapter(completion(BAD_CHANNEL), completion())

    with active_budget(budget), pytest.raises(BudgetExceededError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 1  # the repair would have been a second call
    assert info.value.usage.input_tokens == 100  # the invalid reply was paid for


def test_an_exhausted_time_budget_stops_the_call_before_it_is_made() -> None:
    clock = FakeClock()
    budget = run_budget(clock, max_seconds=90)
    clock.advance(90)
    adapter = ScriptedAdapter(completion())

    with active_budget(budget), pytest.raises(BudgetExceededError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 0
    assert info.value.kind == "time"


def test_a_retry_does_not_start_once_the_backoff_has_used_up_the_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    budget = run_budget(clock, max_seconds=90)

    def sleep_past_the_limit(seconds: float) -> None:
        clock.advance(100)

    monkeypatch.setattr(gateway, "_sleep", sleep_past_the_limit)
    adapter = ScriptedAdapter(TransientProviderError("429", kind="rate_limit"), completion())

    with active_budget(budget), pytest.raises(BudgetExceededError) as info:
        complete_structured("p", Variant, "fast", adapter=adapter, settings=IsolatedSettings())

    assert adapter.call_count == 1  # the second attempt was never made
    assert info.value.kind == "time"


def test_a_budget_that_still_has_room_changes_nothing_about_the_call() -> None:
    adapter = FakeAdapter(completion())
    settings = IsolatedSettings()

    with active_budget(run_budget(FakeClock())):
        variant, usage = complete_structured(
            "write copy", Variant, "strong", adapter=adapter, settings=settings
        )

    assert variant.id == "v1"
    assert usage.input_tokens == 100
    (call,) = adapter.calls
    assert call["timeout_seconds"] == settings.budgets.request_timeout_seconds
