"""LLM gateway: the single door between agents and a model provider (BF-08).

Agents ask for a *tier* and a Pydantic *schema*; they never see a model name, an
SDK object or a raw response. The gateway core resolves the tier to a provider and
model, hands the call to that provider's adapter, then does everything that must be
identical for every provider: cost, validation into the schema and typed errors.

This module imports no provider SDK. Provider specifics live in
`brandforge.llm.adapters`; adding a provider means adding an adapter and a branch
in `brandforge.llm.registry`, not touching this file.

A reply that does not validate gets one repair attempt (BF-21): the call is repeated
with the model's own reply and the validation errors in the prompt. Every call made,
failed ones included, is added to the returned `Usage`.

When a run budget is installed (BF-23, see `brandforge.budget`), the gateway checks it before
every attempt, retries and repairs included, and records the usage of every reply into it. A
budget that is used up raises `BudgetExceededError` before the call is made. With no budget
installed nothing is checked.

Later tasks extend this module, not its callers: tracing and caching (BF-25, BF-27).
"""

import logging
import time
from collections.abc import Callable

from pydantic import BaseModel, ValidationError
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from brandforge.budget import current_budget
from brandforge.config import Settings, Tier, TierPrice, get_settings
from brandforge.llm.base import (
    BudgetExceededError,
    GatewayConfigError,
    GatewayError,
    ProviderAdapter,
    RawCompletion,
    StructuredOutputError,
    TransientProviderError,
)
from brandforge.llm.registry import get_adapter
from brandforge.llm.schema import schema_problems
from brandforge.models import Usage
from brandforge.prompts.loader import load_prompt, render_prompt

__all__ = [
    "BudgetExceededError",
    "GatewayConfigError",
    "GatewayError",
    "ProviderAdapter",
    "RawCompletion",
    "StructuredOutputError",
    "TransientProviderError",
    "complete_structured",
    "schema_problems",
]


logger = logging.getLogger(__name__)


def _sleep(seconds: float) -> None:
    # A named seam so tests can replace the wait without patching `time.sleep` globally.
    time.sleep(seconds)


def _call_with_retries(call: Callable[[], RawCompletion], settings: Settings) -> RawCompletion:
    """Run `call` (one adapter request), retrying transient provider failures with backoff.

    Only `TransientProviderError` is retried: a timeout, dropped connection, rate
    limit or 5xx, as classified by the adapter. Everything else (a refusal, a reply
    that does not validate, a bad request, a missing key) would fail the same way
    again, so it propagates on the first attempt. When the attempts run out the last
    `TransientProviderError` is raised.
    """
    budgets = settings.budgets
    retrying = Retrying(
        retry=retry_if_exception_type(TransientProviderError),
        stop=stop_after_attempt(budgets.max_llm_retries),
        wait=wait_exponential_jitter(
            initial=budgets.retry_initial_wait_seconds, max=budgets.retry_max_wait_seconds
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        sleep=_sleep,
        reraise=True,
    )
    return retrying(call)


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

    Returns the parsed model and the `Usage` (tokens and USD) of every call it took,
    so a repaired reply is reported at the cost of both calls. `adapter` overrides the
    provider lookup; it exists for tests and for callers that manage their own client.

    If a reply comes back whole but does not validate, the gateway asks again, up to
    `budgets.max_schema_repairs` times, with the invalid reply and the validation
    errors in the prompt. A refused or truncated reply is not re-asked.

    Raises:
        GatewayConfigError: a response schema that is not portable (see
            `schema_problems`), a provider with no adapter, or a missing API key.
        StructuredOutputError: the reply was refused, truncated, or still did not
            validate after the repair attempts. Its `raw_text` and `validation_error`
            are those of the last reply, and its `usage` covers every call made.
        TransientProviderError: the provider kept timing out, dropping the connection,
            rate limiting or returning 5xx for `budgets.max_llm_retries` attempts.
        BudgetExceededError: the run's token or wall-clock budget was used up before an
            attempt (the first call, a retry or a repair), so that call was not made. Only
            raised inside a run budget; its `usage` is what the node had spent so far.
        Any other provider SDK error (a 400 or 401, say) passes through unchanged and
        is not retried.
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

    total = Usage()
    next_prompt = prompt
    repairs_left = cfg.budgets.max_schema_repairs
    while True:
        try:
            parsed, usage = _complete_once(
                chosen, ref.model, next_prompt, system, schema, tier, cfg
            )
        except StructuredOutputError as exc:
            total += exc.usage
            # Only a reply that came back whole but invalid is worth re-asking about. A
            # truncated or refused reply has no validation error and would fail the same way.
            if exc.validation_error is None or repairs_left == 0:
                exc.usage = total
                raise
            repairs_left -= 1
            logger.warning(
                "Reply did not validate as %s (%d error(s)); re-asking with the errors.",
                schema.__name__,
                exc.validation_error.error_count(),
            )
            next_prompt = _repair_prompt(prompt, exc.raw_text, exc.validation_error, cfg)
        else:
            return parsed, total + usage


def _complete_once[T: BaseModel](
    adapter: ProviderAdapter,
    model: str,
    prompt: str,
    system: str | None,
    schema: type[T],
    tier: Tier,
    cfg: Settings,
) -> tuple[T, Usage]:
    """One model call, retried on transient failures, validated into `schema`.

    On any unusable reply it raises `StructuredOutputError` carrying the usage of this
    call alone; `complete_structured` adds it to the total.

    Under a run budget, the budget is checked before every attempt, so a retry that follows a
    long backoff cannot start once the time is gone, and the usage of the reply is recorded
    into it before the reply is validated, so a reply that turns out to be unusable still counts.
    """
    budget = current_budget()

    def attempt() -> RawCompletion:
        if budget is not None:
            budget.check()
        return adapter.complete(
            model=model,
            prompt=prompt,
            system=system,
            schema=schema,
            max_output_tokens=cfg.budgets.max_output_tokens_per_call,
            timeout_seconds=cfg.budgets.request_timeout_seconds,
        )

    raw = _call_with_retries(attempt, cfg)
    usage = _usage_from(raw, cfg.pricing.for_tier(tier))
    if budget is not None:
        budget.record(usage)

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


def _describe_validation_error(error: ValidationError) -> str:
    """The validation problems as one line each: where, and what is wrong.

    Offending values are left out: they are in the previous reply the model is shown
    anyway, and some (a whole malformed document) are not useful on a line of their own.
    """
    lines = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        where = ".".join(str(part) for part in item["loc"]) or "(whole reply)"
        lines.append(f"- {where}: {item['msg']}")
    return "\n".join(lines)


def _repair_prompt(
    original_prompt: str, previous_reply: str, error: ValidationError, cfg: Settings
) -> str:
    """The original request plus the model's invalid reply and what was wrong with it."""
    return render_prompt(
        load_prompt("repair", cfg.repair_prompt_version),
        original_prompt=original_prompt,
        previous_reply=previous_reply,
        problems=_describe_validation_error(error),
    )
