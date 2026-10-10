"""Judge rubric for an eval row (BF-36).

promptfoo runs one ``llm-rubric`` per row. This module is the grader behind that
assertion: it scores each variant on its own, through the gateway, on the judge
tier, against the brand rubric's 1-5 anchors. The row's score is the mean. A low
score stays a measurement. A row that cannot be scored fails the assertion. The
rules are in ADR 0029.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from brandforge.brands import BrandLoadError, load_brand
from brandforge.config import Settings, Tier, TierPrice, get_settings
from brandforge.evals.cases import CaseLoadError, EvalCase, load_case
from brandforge.evals.promptfoo import EvalOutput
from brandforge.llm.base import GatewayError, StructuredOutputError
from brandforge.llm.gateway import complete_structured
from brandforge.models import BrandProfile, Brief, Usage, Variant
from brandforge.prompts.loader import PromptParts, load_prompt, render_prompt_parts
from brandforge.scoring import CriterionScore, CriticReply, CritiqueError, build_critique

JUDGE_TIER: Tier = "judge"
_ROLES = frozenset({"judge", "crosscheck"})


class JudgeReply(BaseModel):
    """What the judge returns for one variant: a score per criterion, and nothing else.

    A list of items, never a dict, so the schema stays portable. The judge does not
    suggest edits and does not decide pass or fail. ``build_critique`` checks the
    list against the rubric.
    """

    model_config = ConfigDict(extra="forbid")

    scores: list[CriterionScore]


class StructuredCompleter(Protocol):
    """The slice of ``complete_structured`` this module uses. Tests pass a fake."""

    def __call__(
        self,
        prompt: str,
        schema: type[JudgeReply],
        tier: Tier,
        /,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]: ...


@dataclass(frozen=True)
class VariantJudgement:
    """One variant's scores. ``overall`` is the mean of its criterion scores."""

    variant_id: str
    channel: str
    scores: dict[str, int]
    overall: float


@dataclass(frozen=True)
class JudgeGrade:
    """The grader's result for one row.

    ``passed`` means the row was scored. ``score`` is the mean of the criterion
    means, on the 1-5 scale, or 0 when the row was not scored. ``criteria`` is
    empty on a failure.
    """

    passed: bool
    score: float
    reason: str
    criteria: dict[str, float]
    variants: tuple[VariantJudgement, ...]
    usage: Usage
    model: str
    rubric_version: str
    prompt_version: str
    role: str

    def metadata(self) -> dict[str, Any]:
        """The breakdown promptfoo stores beside the pass, score and reason."""
        return {
            "role": self.role,
            "judge_model": self.model,
            "rubric_version": self.rubric_version,
            "prompt_version": self.prompt_version,
            "criteria": dict(self.criteria),
            "variants": [
                {
                    "id": item.variant_id,
                    "channel": item.channel,
                    "scores": dict(item.scores),
                    "overall": item.overall,
                }
                for item in self.variants
            ],
        }


def rubric_block(brand: BrandProfile) -> str:
    """Every criterion with all five anchors, best first."""
    blocks: list[str] = []
    for criterion in brand.rubric.criteria:
        levels = "\n".join(f"  {level}: {criterion.anchors[level]}" for level in range(5, 0, -1))
        blocks.append(f"- {criterion.name}: {criterion.description}\n{levels}")
    return "\n".join(blocks)


def anchored_rubric(brand: BrandProfile) -> str:
    """The ``llm-rubric`` value promptfoo shows for this brand.

    The model sees the same anchors inside the versioned judge prompt. This string
    is the copy of them on the assertion, so the eval record names the scale.
    """
    return "\n".join(
        (
            "Score each variant on its own, from 1 to 5, against the anchored criteria below.",
            "Score a variant by itself. The row score is the mean across variants.",
            "A finished grade records the mean. It does not accept or reject the row.",
            "",
            rubric_block(brand),
        )
    )


def judge_prompt(
    brief: Brief, brand: BrandProfile, variant: Variant, *, version: str
) -> PromptParts:
    """The judge prompt for one variant. The brand and rubric are the cached prefix."""
    return render_prompt_parts(
        load_prompt("judge", version),
        brand_id=brand.id,
        voice=", ".join(brand.voice),
        do=_bullets(brand.do),
        dont=_bullets(brand.dont),
        banned_words=", ".join(brand.banned_words) or "(none)",
        criteria_names=", ".join(criterion.name for criterion in brand.rubric.criteria),
        rubric=rubric_block(brand),
        channel=variant.channel,
        headline=variant.headline,
        body=variant.body,
        cta=variant.cta,
        product=brief.product,
        objective=brief.objective,
        constraints=_bullets(brief.constraints),
    )


def judge_assertions(
    brand: BrandProfile, *, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """The ``llm-rubric`` assertions for one case.

    The judge tier is always included. A configured cross-check model adds a second
    assertion that grades the same anchors again.
    """
    cfg = settings if settings is not None else get_settings()
    rubric = anchored_rubric(brand)
    assertions = [_assertion(rubric, role="judge", metric="judge")]
    if cfg.judge_crosscheck is not None:
        assertions.append(_assertion(rubric, role="crosscheck", metric="judge_crosscheck"))
    return assertions


def score_row(
    output: object,
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter | None = None,
    role: str = "judge",
) -> JudgeGrade:
    """Score every variant in an eval row. One model call per variant.

    A variant is sent alone, so the judge has no neighbouring copy to compare it
    with. The criterion means are taken across variants, and ``score`` is the mean
    of those means. A score of 1 is still a finished grade.
    """
    cfg = settings if settings is not None else get_settings()
    completer = complete if complete is not None else complete_structured
    loaded = _load_row(_coerce(output))
    if isinstance(loaded, str):
        return _failed(loaded, settings=cfg, role=role)
    row, case, brand = loaded
    if not row.variants:
        return _failed(
            "output has no variants",
            settings=cfg,
            role=role,
            rubric_version=brand.rubric.version,
        )

    names = [criterion.name for criterion in brand.rubric.criteria]
    totals = dict.fromkeys(names, 0)
    judgements: list[VariantJudgement] = []
    usage = Usage()
    for item in row.variants:
        variant = Variant(
            id=item.id,
            channel=item.channel,
            headline=item.headline,
            body=item.body,
            cta=item.cta,
        )
        try:
            prompt = judge_prompt(case.brief, brand, variant, version=cfg.judge_prompt_version)
            reply, call_usage = completer(
                prompt.user, JudgeReply, JUDGE_TIER, system=prompt.system, settings=cfg
            )
        except GatewayError as exc:
            spent = usage
            if isinstance(exc, StructuredOutputError):
                spent = spent + exc.usage
            return _failed(
                f"{variant.id}: {_error(exc)}",
                settings=cfg,
                role=role,
                usage=spent,
                rubric_version=brand.rubric.version,
            )
        usage = usage + call_usage
        try:
            critique = build_critique(
                variant.id,
                CriticReply(scores=reply.scores, fixes=[]),
                brand.rubric,
                cfg.thresholds,
            )
        except CritiqueError as exc:
            return _failed(
                f"{variant.id}: {_error(exc)}",
                settings=cfg,
                role=role,
                usage=usage,
                rubric_version=brand.rubric.version,
            )
        for name in names:
            totals[name] += critique.scores[name]
        judgements.append(
            VariantJudgement(
                variant_id=variant.id,
                channel=variant.channel,
                scores=dict(critique.scores),
                overall=critique.overall,
            )
        )

    count = len(judgements)
    criteria = {name: totals[name] / count for name in names}
    score = sum(criteria.values()) / len(criteria)
    return JudgeGrade(
        passed=True,
        score=score,
        reason=_reason(row, brand, criteria, judgements, score),
        criteria=criteria,
        variants=tuple(judgements),
        usage=usage,
        model=cfg.models.judge,
        rubric_version=brand.rubric.version,
        prompt_version=cfg.judge_prompt_version,
        role=role,
    )


def call_judge(
    prompt: object,
    options: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """promptfoo's grader entry. ``rubricPrompt`` is ``{{output}}``, the eval row.

    When the rendered prompt is not the raw row, ``context["vars"]["output"]`` is
    the fallback. A cross-check role grades with the cross-check model and its
    prices, through the same judge tier the gateway already knows.
    """
    try:
        role = _role(options or {})
        settings = get_settings()
        if role == "crosscheck":
            settings = crosscheck_settings(settings)
        grade = score_row(_row_text(prompt, context), settings=settings, role=role)
        return _provider_result(grade)
    except AssertionError:
        raise
    except Exception as exc:
        return {"output": {"pass": False, "score": 0.0, "reason": _error(exc)}}


def crosscheck_settings(settings: Settings) -> Settings:
    """Settings whose judge tier is the cross-check model, priced for that model.

    The gateway only prices tiers. The cross-check borrows the judge tier for one
    call so the adapter, retries and schema check stay the ones the judge uses.
    The Opus cache-read override is left behind: a copied price uses the
    cross-check multiplier, or the provider default when that multiplier is unset.
    """
    text = settings.judge_crosscheck_model.strip()
    if settings.judge_crosscheck is None or not text:
        raise ValueError(
            "Judge cross-check is off. Set BRANDFORGE_JUDGE_CROSSCHECK_MODEL and its prices."
        )
    price = TierPrice(
        input_per_mtok=settings.judge_crosscheck_input_per_mtok,
        output_per_mtok=settings.judge_crosscheck_output_per_mtok,
        cache_read_multiplier=settings.judge_crosscheck_cache_read_multiplier,
    )
    return settings.model_copy(
        update={
            "models": settings.models.model_copy(update={"judge": text}),
            "pricing": settings.pricing.model_copy(update={"judge": price}),
        }
    )


# promptfoo forces every weight-0 assertion to pass, so a grader refusal would
# not fail the eval. A small weight leaves the 1-5 mean almost out of the 0-1
# average and lets a refusal fail the run (ADR 0029, ADR 0031).
JUDGE_ASSERTION_WEIGHT = 0.001


def _assertion(rubric: str, *, role: str, metric: str) -> dict[str, Any]:
    # The rubric prompt is the row itself. This provider scores it, and the
    # versioned judge prompt is what the model sees.
    return {
        "type": "llm-rubric",
        "value": rubric,
        "metric": metric,
        "weight": JUDGE_ASSERTION_WEIGHT,
        "rubricPrompt": "{{output}}",
        "provider": {
            "id": "file://providers/judge.py",
            "config": {"role": role},
        },
    }


def _load_row(text: str) -> tuple[EvalOutput, EvalCase, BrandProfile] | str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"output is not valid JSON ({exc.msg})"
    try:
        row = EvalOutput.model_validate(payload)
    except ValidationError as exc:
        return f"output is not the eval row: {_schema_reason(exc)}"
    try:
        case = load_case(row.case_id)
    except CaseLoadError as exc:
        return _error(exc)
    if row.brand_id != case.brand_id:
        return f"row brand_id is {row.brand_id!r}; this case is {case.brand_id!r}"
    try:
        brand = load_brand(case.brand_id)
    except BrandLoadError as exc:
        return _error(exc)
    return row, case, brand


def _reason(
    row: EvalOutput,
    brand: BrandProfile,
    criteria: dict[str, float],
    judgements: list[VariantJudgement],
    score: float,
) -> str:
    parts = ", ".join(f"{name} {value:.2f}" for name, value in criteria.items())
    lines = [f"mean {score:.2f} ({parts})"]
    if row.rubric_version != brand.rubric.version:
        lines.append(
            f"The row records rubric {row.rubric_version}; "
            f"this score uses rubric {brand.rubric.version} on disk."
        )
    for item in judgements:
        bits = ", ".join(f"{name} {item.scores[name]}" for name in criteria)
        lines.append(f"{item.variant_id} {item.channel}: {bits}")
    return "\n".join(lines)


def _failed(
    reason: str,
    *,
    settings: Settings,
    role: str,
    usage: Usage | None = None,
    rubric_version: str = "",
) -> JudgeGrade:
    return JudgeGrade(
        passed=False,
        score=0.0,
        reason=reason,
        criteria={},
        variants=(),
        usage=usage if usage is not None else Usage(),
        model=settings.models.judge,
        rubric_version=rubric_version,
        prompt_version=settings.judge_prompt_version,
        role=role,
    )


def _provider_result(grade: JudgeGrade) -> dict[str, Any]:
    result: dict[str, Any] = {
        "output": {"pass": grade.passed, "score": grade.score, "reason": grade.reason},
        "metadata": grade.metadata(),
    }
    if grade.usage.total_tokens or grade.usage.cost_usd:
        result["tokenUsage"] = {
            "prompt": (
                grade.usage.input_tokens
                + grade.usage.cache_write_tokens
                + grade.usage.cache_read_tokens
            ),
            "completion": grade.usage.output_tokens,
            "total": grade.usage.total_tokens,
        }
        result["cost"] = grade.usage.cost_usd
    return result


def _row_text(prompt: object, context: Mapping[str, Any] | None) -> str:
    if isinstance(prompt, str) and prompt.lstrip().startswith("{"):
        return prompt
    if context is not None:
        variables = context.get("vars")
        if isinstance(variables, Mapping) and "output" in variables:
            return _coerce(variables["output"])
    return _coerce(prompt)


def _role(options: Mapping[str, Any]) -> str:
    raw = options.get("config")
    config = raw if isinstance(raw, Mapping) else {}
    role = config.get("role", "judge")
    if not isinstance(role, str) or role not in _ROLES:
        raise ValueError(f"judge role must be 'judge' or 'crosscheck', got {role!r}")
    return role


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def _coerce(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _schema_reason(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        message = str(err["msg"])
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) if parts else "schema validation failed"


def _error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
