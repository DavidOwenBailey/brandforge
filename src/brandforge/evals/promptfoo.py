"""promptfoo providers for the baseline and the pipeline (BF-34).

promptfoo is the runner. It does not write the model prompt: each test carries a
case id, and the provider loads that case and calls the system that already owns
the versioned prompt. Both systems return one JSON row so the deterministic
assertions can score them side by side. See ADR 0027 and ADR 0028.
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from brandforge.evals.cases import EvalCase, load_case, load_cases
from brandforge.models import (
    BrandProfile,
    Channel,
    FinalStatus,
    NonEmptyStr,
    RunResult,
    Usage,
    Variant,
)

System = Literal["baseline", "pipeline"]


class _Row(BaseModel):
    """Base for the eval row: unknown fields are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


class EvalVariant(_Row):
    """One variant as the eval sees it.

    ``flagged`` is null for the baseline, which has no critic. The pipeline
    copies the assembler's flag.
    """

    id: NonEmptyStr
    channel: Channel
    headline: NonEmptyStr
    body: str
    cta: NonEmptyStr
    flagged: bool | None


class EvalUsage(_Row):
    """Tokens and USD for the row. ``total_tokens`` counts cached tokens too."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _total_matches_the_parts(self) -> "EvalUsage":
        parts = (
            self.input_tokens
            + self.output_tokens
            + self.cache_write_tokens
            + self.cache_read_tokens
        )
        if self.total_tokens != parts:
            raise ValueError(f"total_tokens is {self.total_tokens}; the parts add up to {parts}")
        return self


class EvalOutput(_Row):
    """The JSON both providers return.

    ``run_id`` and ``trace_id`` are null for the baseline, which is one call
    and not a graph run. A pipeline row copies them from the ``RunResult``.
    """

    system: System
    case_id: NonEmptyStr
    brand_id: NonEmptyStr
    brand_version: NonEmptyStr
    rubric_version: NonEmptyStr
    status: FinalStatus
    run_id: NonEmptyStr | None = None
    trace_id: str | None = None
    revision_count: int = Field(ge=0)
    variants: list[EvalVariant]
    errors: list[str]
    usage: EvalUsage


def generate_tests(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """One promptfoo test per eval case, in case-id order.

    ``config["case_ids"]``, when set, keeps only those ids and keeps the order
    given. The committed config omits it, so ``promptfoo eval`` runs the full set.
    """
    loaded = load_cases()
    by_id = {case.id: case for case in loaded}
    selected = _selected_ids(config)
    if selected is None:
        ordered = loaded
    else:
        missing = [case_id for case_id in selected if case_id not in by_id]
        if missing:
            known = ", ".join(sorted(by_id))
            raise ValueError(f"Unknown case(s): {', '.join(missing)}. Known cases: {known}")
        ordered = [by_id[case_id] for case_id in selected]
    return [_test(case) for case in ordered]


def call_baseline(
    prompt: str, options: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one case through the single-prompt baseline.

    ``options`` is promptfoo's provider config. The worker reads it (timeout,
    the Python executable). This function does not.
    """
    del options
    try:
        case, brand = _load(prompt, context)
        from brandforge.baseline import generate_baseline

        variants, usage = generate_baseline(case.brief, brand)
        output = _output(
            "baseline",
            case,
            brand_id=brand.id,
            brand_version=brand.version,
            rubric_version=brand.rubric.version,
            status="complete",
            variants=_from_variants(variants, flagged=None),
            usage=usage,
        )
        return _ok(output, requests=1)
    except Exception as exc:
        return _fail(exc)


def call_pipeline(
    prompt: str, options: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one case through the full graph.

    No checkpointer: an eval row is the result, not a resumable run. Tracing
    follows settings, as it does for the CLI.
    """
    del options
    try:
        case, brand = _load(prompt, context)
        from brandforge.graph import run_graph

        state = run_graph(case.brief, brand)
        result = state["result"]
        if result is None:
            raise RuntimeError("the pipeline finished without a result")
        output = _from_result(case, result)
        # Node usage is summed across revision passes, so a request count taken
        # from it would be short. The token totals and the cost are exact.
        return _ok(output, requests=None)
    except Exception as exc:
        return _fail(exc)


def _test(case: EvalCase) -> dict[str, Any]:
    return {
        "description": case.id,
        "vars": {"case_id": case.id},
        "metadata": {"brand_id": case.brand_id},
    }


def _selected_ids(config: Mapping[str, Any] | None) -> list[str] | None:
    if config is None or "case_ids" not in config:
        return None
    raw = config["case_ids"]
    if not isinstance(raw, list):
        raise ValueError("case_ids must be a list of case id strings")
    ids: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("case_ids must be a list of case id strings")
        ids.append(item.strip())
    return ids


def _load(prompt: object, context: object) -> tuple[EvalCase, BrandProfile]:
    from brandforge.brands import load_brand

    case = load_case(case_id_from_context(prompt, context))
    return case, load_brand(case.brand_id)


def case_id_from_context(prompt: object, context: object) -> str:
    """The case to run. Test vars win; the rendered ``{{case_id}}`` prompt is the fallback."""
    raw = _mapping(_mapping(context).get("vars")).get("case_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    raise ValueError(
        "promptfoo test has no case_id. Set vars.case_id or use the {{case_id}} prompt."
    )


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _from_variants(variants: list[Variant], *, flagged: bool | None) -> list[EvalVariant]:
    return [
        EvalVariant(
            id=variant.id,
            channel=variant.channel,
            headline=variant.headline,
            body=variant.body,
            cta=variant.cta,
            flagged=flagged,
        )
        for variant in variants
    ]


def _from_result(case: EvalCase, result: RunResult) -> EvalOutput:
    return _output(
        "pipeline",
        case,
        brand_id=result.brand_id,
        brand_version=result.brand_version,
        rubric_version=result.rubric_version,
        status=result.status,
        run_id=result.run_id,
        trace_id=result.trace_id,
        revision_count=result.revision_count,
        variants=[
            EvalVariant(
                id=item.variant.id,
                channel=item.variant.channel,
                headline=item.variant.headline,
                body=item.variant.body,
                cta=item.variant.cta,
                flagged=item.flagged,
            )
            for item in result.variants
        ],
        errors=[f"{item.node}: {item.message}" for item in result.errors],
        usage=result.usage,
    )


def _output(
    system: System,
    case: EvalCase,
    *,
    brand_id: str,
    brand_version: str,
    rubric_version: str,
    status: FinalStatus,
    variants: list[EvalVariant],
    usage: Usage,
    run_id: str | None = None,
    trace_id: str | None = None,
    revision_count: int = 0,
    errors: list[str] | None = None,
) -> EvalOutput:
    return EvalOutput(
        system=system,
        case_id=case.id,
        brand_id=brand_id,
        brand_version=brand_version,
        rubric_version=rubric_version,
        status=status,
        run_id=run_id,
        trace_id=trace_id,
        revision_count=revision_count,
        variants=variants,
        errors=[] if errors is None else errors,
        usage=EvalUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
        ),
    )


def _token_usage(usage: EvalUsage, *, requests: int | None) -> dict[str, int]:
    """promptfoo's token block. Prompt tokens include the cache, as billed."""
    reported = {
        "total": usage.total_tokens,
        "prompt": usage.input_tokens + usage.cache_write_tokens + usage.cache_read_tokens,
        "completion": usage.output_tokens,
    }
    if requests is not None:
        reported["numRequests"] = requests
    return reported


def _ok(output: EvalOutput, *, requests: int | None) -> dict[str, Any]:
    return {
        "output": output.model_dump_json(),
        "tokenUsage": _token_usage(output.usage, requests=requests),
        "cost": output.usage.cost_usd,
    }


def _fail(exc: Exception) -> dict[str, Any]:
    """A bad case is one failed row. The other cases, and the other provider, still run.

    ``AssertionError`` is a bug in the caller, not a failed generation, so it propagates.
    """
    if isinstance(exc, AssertionError):
        raise exc
    text = " ".join(str(exc).split())
    message = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return {"error": message}
