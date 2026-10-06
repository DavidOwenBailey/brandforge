"""LLM gateway: the single door between agents and a model provider (BF-08).

Agents ask for a *tier* and a Pydantic *schema*; they never see a model name, an
SDK object or a raw response. The gateway core resolves the tier to a provider and
model, hands the call to that provider's adapter, then does everything that must be
identical for every provider: cost, validation into the schema and typed errors.

This module imports no provider SDK. Provider specifics live in
`brandforge.llm.adapters`; adding a provider means adding an adapter and a branch
in `brandforge.llm.registry`, not touching this file.

Later tasks extend this module, not its callers: retries with backoff (BF-20),
schema-repair re-ask (BF-21), budget checks (BF-23), tracing and caching (BF-25, BF-27).
"""

from pydantic import BaseModel, ValidationError

from brandforge.config import Settings, Tier, TierPrice, get_settings
from brandforge.llm.base import (
    GatewayConfigError,
    GatewayError,
    ProviderAdapter,
    RawCompletion,
    StructuredOutputError,
)
from brandforge.llm.registry import get_adapter
from brandforge.llm.schema import schema_problems
from brandforge.models import Usage

__all__ = [
    "GatewayConfigError",
    "GatewayError",
    "ProviderAdapter",
    "RawCompletion",
    "StructuredOutputError",
    "complete_structured",
    "schema_problems",
]


def _usage_from(raw: RawCompletion, price: TierPrice) -> Usage:
    cost = (raw.input_tokens * price.input_per_mtok + raw.output_tokens * price.output_per_mtok) / (
        1_000_000
    )
    return Usage(input_tokens=raw.input_tokens, output_tokens=raw.output_tokens, cost_usd=cost)


def complete_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    tier: Tier,
    *,
    system: str | None = None,
    adapter: ProviderAdapter | None = None,
    settings: Settings | None = None,
) -> tuple[T, Usage]:
    """Send `prompt` to the model for `tier` and return a validated `schema` instance.

    Returns the parsed model and the `Usage` (tokens and USD) for this one call.
    `adapter` overrides the provider lookup; it exists for tests and for callers that
    manage their own client.

    Raises:
        GatewayConfigError: a response schema that is not portable (see
            `schema_problems`), a provider with no adapter, or a missing API key.
        StructuredOutputError: the reply was refused, truncated, or did not validate.
        Provider SDK errors pass through unchanged for now; BF-20 maps them to
        gateway errors inside each adapter and adds the retry policy.
    """
    cfg = settings or get_settings()
    ref = cfg.models.resolve(tier)

    problems = schema_problems(schema)
    if problems:
        raise GatewayConfigError(
            f"{schema.__name__} cannot be used as a response schema: free-form object "
            f"field(s) at {', '.join(problems)}. "
            "Use a list of items instead and convert it in code."
        )

    chosen = adapter or get_adapter(ref.provider, cfg)
    raw = chosen.complete(
        model=ref.model,
        prompt=prompt,
        system=system,
        schema=schema,
        max_output_tokens=cfg.budgets.max_output_tokens_per_call,
        timeout_seconds=cfg.budgets.request_timeout_seconds,
    )
    usage = _usage_from(raw, cfg.pricing.for_tier(tier))

    if raw.outcome != "complete":
        raise StructuredOutputError(
            f"Model stopped early ({raw.outcome}); output is unusable.",
            raw_text=raw.text,
            usage=usage,
            outcome=raw.outcome,
        )

    try:
        parsed = schema.model_validate_json(raw.text)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"Model output did not validate as {schema.__name__}: {exc.error_count()} error(s).",
            raw_text=raw.text,
            usage=usage,
            outcome=raw.outcome,
            validation_error=exc,
        ) from exc
    return parsed, usage
