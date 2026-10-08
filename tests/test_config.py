import pytest
from pydantic_settings import SettingsConfigDict

from brandforge.config import ModelRef, Settings, parse_model_ref


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def test_defaults_match_architecture() -> None:
    s = IsolatedSettings()
    assert s.budgets.max_revisions == 2
    assert s.budgets.max_schema_repairs == 1
    assert s.thresholds.min_overall == 4.0
    assert s.thresholds.min_per_criterion == 3
    assert s.models.fast == "anthropic:claude-haiku-4-5-20251001"
    assert s.planner_prompt_version == "v2"
    assert s.pricing.anthropic_cache.write_multiplier == 1.25
    assert s.pricing.anthropic_cache.read_multiplier == 0.1
    assert s.pricing.gemini_cache.read_multiplier == 0.1
    assert s.pricing.strong.cache_read_multiplier == 0.05
    assert s.pricing.fast.cache_read_multiplier is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRANDFORGE_BUDGETS__MAX_REVISIONS", "1")
    monkeypatch.setenv("BRANDFORGE_MODELS__FAST", "google:test-model")
    s = IsolatedSettings()
    assert s.budgets.max_revisions == 1
    assert s.models.resolve("fast") == ModelRef("google", "test-model")


def test_resolve_default_tiers() -> None:
    s = IsolatedSettings()
    assert s.models.resolve("strong") == ModelRef("anthropic", "claude-sonnet-5-5")


@pytest.mark.parametrize("bad", ["claude-haiku", ":model", "anthropic:", ""])
def test_malformed_model_ref_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_model_ref(bad)


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    s = IsolatedSettings()
    assert s.anthropic_api_key.get_secret_value() == "sk-secret-value"
    assert "sk-secret-value" not in repr(s)


def test_invalid_budget_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRANDFORGE_BUDGETS__MAX_TOKENS_PER_RUN", "0")
    with pytest.raises(ValueError):
        IsolatedSettings()


def test_log_level_and_format_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRANDFORGE_LOG_LEVEL", "debug")
    monkeypatch.setenv("BRANDFORGE_LOG_FORMAT", "Console")
    settings = IsolatedSettings()
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "console"


def test_bad_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRANDFORGE_LOG_LEVEL", "verbose")
    with pytest.raises(ValueError):
        IsolatedSettings()
