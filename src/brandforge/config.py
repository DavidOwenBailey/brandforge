"""Application settings, loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

from pydantic import AfterValidator, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRef(NamedTuple):
    """A parsed 'provider:model' reference."""

    provider: str
    model: str


def parse_model_ref(value: str) -> ModelRef:
    """Split 'provider:model' (e.g. 'anthropic:claude-haiku-4-5-20251001')."""
    provider, sep, model = value.partition(":")
    if not sep or not provider.strip() or not model.strip():
        raise ValueError(f"Expected 'provider:model', got {value!r}")
    return ModelRef(provider.strip().lower(), model.strip())


def _check_model_ref(value: str) -> str:
    parse_model_ref(value)  # raises ValueError if malformed
    return value


ModelId = Annotated[str, AfterValidator(_check_model_ref)]

Tier = Literal["strong", "fast", "judge"]


class ModelTiers(BaseModel):
    """Tier -> 'provider:model'. Agents ask for a tier, never a model name."""

    strong: ModelId = "anthropic:claude-sonnet-5-5"
    fast: ModelId = "anthropic:claude-haiku-4-5-20251001"
    judge: ModelId = "anthropic:claude-opus-5-5"

    def resolve(self, tier: Tier) -> ModelRef:
        return parse_model_ref(getattr(self, tier))


class TierPrice(BaseModel):
    """USD per million tokens for the model currently assigned to a tier."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class Pricing(BaseModel):
    """Tier -> price. Change a tier's model and its price together.

    Defaults match Anthropic's published base prices for the default tiers.
    Cache read/write pricing is out of scope until prompt caching lands (BF-27).
    """

    strong: TierPrice = TierPrice(input_per_mtok=2.0, output_per_mtok=10.0)
    fast: TierPrice = TierPrice(input_per_mtok=1.0, output_per_mtok=5.0)
    judge: TierPrice = TierPrice(input_per_mtok=4.0, output_per_mtok=20.0)

    def for_tier(self, tier: Tier) -> TierPrice:
        price: TierPrice = getattr(self, tier)
        return price


class Budgets(BaseModel):
    """Hard limits so every run is bounded."""

    max_tokens_per_run: int = Field(default=60_000, gt=0)
    max_wall_clock_seconds: int = Field(default=90, gt=0)
    max_revisions: int = Field(default=2, ge=0)
    # Total attempts per model call (the first try included) when the provider fails
    # transiently. 3 means one call plus at most two retries.
    max_llm_retries: int = Field(default=3, ge=1)
    # Backoff between attempts: initial * 2**(n-1) seconds plus up to 1s of jitter, capped.
    retry_initial_wait_seconds: float = Field(default=1.0, ge=0)
    retry_max_wait_seconds: float = Field(default=20.0, ge=0)
    max_output_tokens_per_call: int = Field(default=2_048, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)


class Thresholds(BaseModel):
    """Pass rules the router applies to critiques (1-5 scale)."""

    min_overall: float = Field(default=4.0, ge=1, le=5)
    min_per_criterion: int = Field(default=3, ge=1, le=5)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BRANDFORGE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Secrets (never logged or committed)
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="ANTHROPIC_API_KEY"
    )
    gemini_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GEMINI_API_KEY")
    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    langfuse_public_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    # Behaviour
    models: ModelTiers = ModelTiers()
    pricing: Pricing = Pricing()
    budgets: Budgets = Budgets()
    thresholds: Thresholds = Thresholds()
    retrieval_enabled: bool = True  # used by BF-32
    baseline_prompt_version: str = "v1"  # prompts/baseline_<version>.md
    baseline_variants_per_channel: int = Field(default=3, ge=1, le=10)
    planner_prompt_version: str = "v1"  # prompts/planner_<version>.md
    planner_max_variants_per_channel: int = Field(default=5, ge=1, le=10)
    writer_prompt_version: str = "v1"  # prompts/writer_<version>.md
    critic_prompt_version: str = "v1"  # prompts/critic_<version>.md
    reviser_prompt_version: str = "v1"  # prompts/reviser_<version>.md

    # Paths
    brands_dir: Path = Path("src/brandforge/brands")
    checkpoint_db: Path = Path(".brandforge/checkpoints.sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
