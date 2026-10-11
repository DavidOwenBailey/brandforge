"""Application settings, loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, NamedTuple
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, SecretStr, model_validator
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


def _strip_upper(value: object) -> object:
    return value.strip().upper() if isinstance(value, str) else value


def _strip_lower(value: object) -> object:
    return value.strip().lower() if isinstance(value, str) else value


LogLevel = Annotated[
    Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    BeforeValidator(_strip_upper),
]
LogFormat = Annotated[Literal["json", "console"], BeforeValidator(_strip_lower)]


def normalize_api_base_url(value: str) -> str:
    """An http(s) origin the demo page can call, with no path, query or fragment.

    A trailing slash is removed so joining ``/generate`` does not produce a double slash.
    """
    text = value.strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_base_url must be an http or https URL with a host")
    if parsed.path not in {"", "/"}:
        raise ValueError("api_base_url must not include a path")
    if parsed.query or parsed.fragment:
        raise ValueError("api_base_url must not include a query or a fragment")
    return text


ApiBaseUrl = Annotated[str, AfterValidator(normalize_api_base_url)]


class ModelTiers(BaseModel):
    """Tier -> 'provider:model'. Agents ask for a tier, never a model name."""

    strong: ModelId = "anthropic:claude-sonnet-5-5"
    fast: ModelId = "anthropic:claude-haiku-4-5-20251001"
    judge: ModelId = "anthropic:claude-opus-5-5"

    def resolve(self, tier: Tier) -> ModelRef:
        return parse_model_ref(getattr(self, tier))


class TierPrice(BaseModel):
    """USD per million tokens for the model currently assigned to a tier.

    Cache multipliers override the provider's rates for this tier only. Leave them unset to
    use the provider default. Sonnet 5.5 and Opus 5.5 publish a 0.05x cache read, which is
    not Anthropic's 0.1x default, so the strong and judge tiers set the read override.
    """

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cache_write_multiplier: float | None = Field(default=None, ge=0)
    cache_read_multiplier: float | None = Field(default=None, ge=0)


class CacheRates(BaseModel):
    """Multipliers on a tier's input price for prompt-cache writes and reads (BF-27)."""

    write_multiplier: float = Field(ge=0)
    read_multiplier: float = Field(ge=0)


class Pricing(BaseModel):
    """Tier prices plus one cache rate per provider. Change a tier's model and its price together.

    Defaults match the published base prices for the default tiers. Cache rates are per provider
    because Anthropic and Gemini do not bill a cached prefix the same way (ADR 0021):

    - Anthropic's 5-minute ephemeral cache: writes at 1.25x the input price, reads at 0.1x.
    - Gemini implicit caching: a hit is reported as cached input and billed at 0.1x (a 90%
      discount). A miss is ordinary input, so there is no write premium to apply.

    A tier's `cache_read_multiplier` or `cache_write_multiplier`, when set, replaces the
    provider rate for that side. An unknown provider is billed at 1x for both, which is the
    full input price.
    """

    strong: TierPrice = TierPrice(
        input_per_mtok=2.0, output_per_mtok=10.0, cache_read_multiplier=0.05
    )
    fast: TierPrice = TierPrice(input_per_mtok=1.0, output_per_mtok=5.0)
    judge: TierPrice = TierPrice(
        input_per_mtok=4.0, output_per_mtok=20.0, cache_read_multiplier=0.05
    )
    anthropic_cache: CacheRates = CacheRates(write_multiplier=1.25, read_multiplier=0.1)
    gemini_cache: CacheRates = CacheRates(write_multiplier=1.0, read_multiplier=0.1)

    def for_tier(self, tier: Tier) -> TierPrice:
        price: TierPrice = getattr(self, tier)
        return price

    def rates_for(self, provider: str, tier: Tier) -> CacheRates:
        """The cache multipliers for this call: the provider's, unless the tier overrides one."""
        if provider == "anthropic":
            base = self.anthropic_cache
        elif provider == "gemini":
            base = self.gemini_cache
        else:
            base = CacheRates(write_multiplier=1.0, read_multiplier=1.0)
        price = self.for_tier(tier)
        return CacheRates(
            write_multiplier=(
                base.write_multiplier
                if price.cache_write_multiplier is None
                else price.cache_write_multiplier
            ),
            read_multiplier=(
                base.read_multiplier
                if price.cache_read_multiplier is None
                else price.cache_read_multiplier
            ),
        )


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
    # Extra calls allowed per model call when the reply does not validate against the
    # schema: each re-asks with the validation errors in the prompt. 1 means one repair
    # attempt; 0 turns the repair retry off.
    max_schema_repairs: int = Field(default=1, ge=0)
    max_output_tokens_per_call: int = Field(default=2_048, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)


class Thresholds(BaseModel):
    """Pass rules the router applies to critiques (1-5 scale)."""

    min_overall: float = Field(default=4.0, ge=1, le=5)
    min_per_criterion: int = Field(default=3, ge=1, le=5)


# Langfuse Cloud's free tier is the default backend (BF-28, ADR 0022). The optional
# self-hosted stack serves LANGFUSE_LOCAL_HOST and is unused unless LANGFUSE_HOST points at it.
LANGFUSE_CLOUD_HOST = "https://cloud.langfuse.com"
LANGFUSE_LOCAL_HOST = "http://localhost:3000"


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
    langfuse_host: str = Field(default=LANGFUSE_CLOUD_HOST, validation_alias="LANGFUSE_HOST")

    # Behaviour
    # Master switch for Langfuse tracing (BF-25). Tracing also needs both Langfuse keys: with
    # either one missing it stays off, so a fresh clone with no keys runs exactly as before.
    tracing_enabled: bool = True
    # Structured logs (BF-26). JSON lines on stderr; `console` is the same fields for a person.
    # The level is read case-insensitively. See `brandforge.logging`.
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "json"
    # HTTP API (BF-39, ADR 0032). Loopback by default: the route has no auth.
    # `brandforge serve` binds here. Set the host only when you mean to expose the port.
    api_host: str = Field(default="127.0.0.1", min_length=1)
    api_port: int = Field(default=8000, ge=1, le=65535)
    # Demo page (BF-40, ADR 0033). Loopback by default, for the same reason as the API.
    # `brandforge demo` binds here. `api_base_url` is the origin the page posts briefs to.
    # A bind host such as 0.0.0.0 is not a URL a client can call, so the two stay separate.
    demo_host: str = Field(default="127.0.0.1", min_length=1)
    demo_port: int = Field(default=8501, ge=1, le=65535)
    api_base_url: ApiBaseUrl = "http://127.0.0.1:8000"
    models: ModelTiers = ModelTiers()
    pricing: Pricing = Pricing()
    budgets: Budgets = Budgets()
    thresholds: Thresholds = Thresholds()
    # Master switch for the retriever (BF-32, ADR 0025). Off writes an empty example list
    # and does not search, so evals can compare the pipeline with and without examples.
    # `brandforge index` ignores this flag (ADR 0023).
    retrieval_enabled: bool = True
    # Nearest approved examples the retriever fetches for each channel in the plan (BF-31).
    # The architecture asks for the 3-5 nearest. See ADR 0024.
    retrieval_examples_per_channel: int = Field(default=4, ge=1, le=5)
    # v2 splits the static prefix (instructions and brand profile) off for prompt caching.
    # v1 is the same words as one message, with no cache break. The repair prompt has no
    # static prefix of its own: a repair keeps the system prompt of the call it corrects.
    baseline_prompt_version: str = "v2"  # prompts/baseline_<version>.md
    baseline_variants_per_channel: int = Field(default=3, ge=1, le=10)
    planner_prompt_version: str = "v2"  # prompts/planner_<version>.md
    planner_max_variants_per_channel: int = Field(default=5, ge=1, le=10)
    writer_prompt_version: str = "v2"  # prompts/writer_<version>.md
    critic_prompt_version: str = "v2"  # prompts/critic_<version>.md
    reviser_prompt_version: str = "v2"  # prompts/reviser_<version>.md
    repair_prompt_version: str = "v1"  # prompts/repair_<version>.md (gateway schema repair)
    judge_prompt_version: str = "v1"  # prompts/judge_<version>.md
    # Optional second judge (BF-36, ADR 0029). Blank leaves scoring on the judge tier
    # alone. A provider:model value cross-checks the same rubric; its prices replace
    # the judge tier's for that call, so an Opus rate is not applied to a Gemini model.
    judge_crosscheck_model: str = ""
    judge_crosscheck_input_per_mtok: float = Field(default=0.0, ge=0)
    judge_crosscheck_output_per_mtok: float = Field(default=0.0, ge=0)
    judge_crosscheck_cache_read_multiplier: float | None = Field(default=None, ge=0)

    # Paths
    brands_dir: Path = Path("src/brandforge/brands")
    checkpoint_db: Path = Path(".brandforge/checkpoints.sqlite")
    # Persistent Chroma directory for `brandforge index` (BF-30, ADR 0023). One collection
    # per brand. Git-ignored with the rest of `.brandforge/`.
    chroma_dir: Path = Path(".brandforge/chroma")

    @model_validator(mode="after")
    def _price_the_crosscheck(self) -> "Settings":
        text = self.judge_crosscheck_model.strip()
        if not text:
            return self
        parse_model_ref(text)
        if self.judge_crosscheck_input_per_mtok <= 0 or self.judge_crosscheck_output_per_mtok <= 0:
            raise ValueError("A judge cross-check model needs input and output prices above 0.")
        return self

    @property
    def judge_crosscheck(self) -> ModelRef | None:
        """The optional second judge, or None when the eval uses the judge tier only."""
        text = self.judge_crosscheck_model.strip()
        if not text:
            return None
        return parse_model_ref(text)

    @property
    def langfuse_configured(self) -> bool:
        """True when both Langfuse keys are set, which is what tracing needs besides the switch."""
        return bool(
            self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key.get_secret_value()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
