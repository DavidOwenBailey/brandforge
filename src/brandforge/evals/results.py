"""Summaries of a promptfoo eval export (BF-38).

The full 30-case run is manual. ``summarize_path`` reads the JSON promptfoo
writes with ``--output`` and reports the baseline and the pipeline side by
side: judge means, a pass rate on the critic's thresholds, the deterministic
checks, flagged variants, tokens, cost and latency. The smoke set named here
is the five cases CI runs. See ADR 0031.
"""

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from brandforge.config import Settings, get_settings
from brandforge.evals.cases import CaseLoadError, list_case_ids, load_case
from brandforge.evals.promptfoo import EvalOutput

# The CI smoke eval. The same ids are pinned in evals/promptfooconfig.smoke.yaml.
# Tests keep the two lists the same. The set covers every brand, every channel,
# a headline cap, a case with no cap, and a required phrase.
SMOKE_CASE_IDS: tuple[str, ...] = (
    "brightleaf_01_spring_blossom",
    "brightleaf_02_starter_box",
    "ledgerly_01_vat_reminders",
    "voltride_01_commuter_ebike",
    "voltride_04_test_ride_weekends",
)

DETERMINISTIC_METRICS: tuple[str, ...] = (
    "valid_json",
    "eval_row",
    "channel_limits",
    "banned_words",
    "cta_present",
    "must_mention",
)

_SYSTEMS: tuple[str, ...] = ("baseline", "pipeline")
_MEAN_LINE = re.compile(r"^mean\s+[0-9]+(?:\.[0-9]+)?\s+\(([^)]*)\)")
_CRITERION = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s+([0-9]+(?:\.[0-9]+)?)")
_BANDS: tuple[str, ...] = ("1-2", "2-3", "3-4", "4-5")

SummaryKind = Literal["full", "smoke", "slice"]


class ResultsError(Exception):
    """The promptfoo export cannot be read or has no rows to summarize."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SummaryThresholds(_Model):
    """The critic thresholds used to decide the reported judge pass rate."""

    min_overall: float
    min_per_criterion: int


class SummaryEnvironment(_Model):
    """Settings in the process that wrote the summary.

    A promptfoo export does not record which model served each tier. Summarize
    in the same environment that ran the eval.
    """

    strong_model: str
    fast_model: str
    judge_model: str
    retrieval_enabled: bool
    max_tokens_per_run: int
    max_wall_clock_seconds: int
    max_revisions: int
    max_llm_retries: int
    retry_initial_wait_seconds: float
    retry_max_wait_seconds: float
    baseline_prompt: str
    planner_prompt: str
    writer_prompt: str
    critic_prompt: str
    reviser_prompt: str
    judge_prompt: str


class RowSummary(_Model):
    """One case through one system.

    ``flagged_variants`` is null when the row has no critic flag, which is every
    baseline row. ``judge_score`` is null when the judge did not finish a grade.
    A score of 0 from a failed grade is not stored as a mean of 0.
    """

    case_id: str
    system: Literal["baseline", "pipeline"]
    brand_id: str | None
    status: str
    deterministic_pass: bool
    deterministic: dict[str, bool]
    judge_score: float | None
    judge_criteria: dict[str, float]
    judge_error: str | None
    crosscheck_score: float | None
    flagged_variants: int | None
    variant_count: int
    tokens: int | None
    cost_usd: float | None
    latency_ms: float | None
    error: str | None


class SystemSummary(_Model):
    """Aggregates for one system. Means count each brief once."""

    rows: int
    errors: int
    incomplete: int
    deterministic_passes: int
    judge_scored: int
    judge_compared: int
    judge_passes: int
    judge_mean: float | None
    judge_criteria: dict[str, float]
    judge_pass_rate: float | None
    judge_min: float | None
    judge_median: float | None
    judge_max: float | None
    judge_bands: dict[str, int]
    crosscheck_mean: float | None
    flagged_variants: int | None
    variants: int | None
    flagged_rate: float | None
    total_tokens: int
    total_cost_usd: float
    mean_tokens: float | None
    mean_cost_usd: float | None
    mean_latency_ms: float | None


class EvalSummary(_Model):
    """The committed shape of one eval run."""

    kind: SummaryKind
    eval_id: str | None
    eval_timestamp: str | None
    summarized_at: str
    case_ids: list[str]
    thresholds: SummaryThresholds
    environment: SummaryEnvironment
    baseline: SystemSummary
    pipeline: SystemSummary
    rows: list[RowSummary]
    warnings: list[str] = Field(default_factory=list)


def summarize_path(path: Path, *, settings: Settings | None = None) -> EvalSummary:
    """Read a promptfoo JSON export and summarize it. Does not call a model."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultsError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResultsError(f"{path} is not JSON ({exc.msg})") from exc
    return summarize_export(payload, settings=settings)


def summarize_export(payload: object, *, settings: Settings | None = None) -> EvalSummary:
    """Summarize a parsed promptfoo export.

    Accepts the ``--output file.json`` wrapper (``results.results``) and a bare
    list of result rows.
    """
    cfg = settings if settings is not None else get_settings()
    raw_rows, eval_id, timestamp = _export_rows(payload)
    if not raw_rows:
        raise ResultsError("promptfoo export has no result rows")

    warnings: list[str] = []
    parsed: dict[tuple[str, str], RowSummary] = {}
    order: list[str] = []
    for index, raw in enumerate(raw_rows, start=1):
        row = _row(raw, index)
        key = (row.system, row.case_id)
        if key in parsed:
            warnings.append(f"duplicate row for {row.system} {row.case_id}; kept the later one")
        else:
            if row.case_id not in order:
                order.append(row.case_id)
        parsed[key] = row

    rows = [_ordered(parsed, case_id) for case_id in _case_order(order)]
    flat = [row for case_rows in rows for row in case_rows]
    thresholds = SummaryThresholds(
        min_overall=cfg.thresholds.min_overall,
        min_per_criterion=cfg.thresholds.min_per_criterion,
    )
    return EvalSummary(
        kind=_kind(order),
        eval_id=eval_id,
        eval_timestamp=timestamp,
        summarized_at=datetime.now(UTC).isoformat(timespec="seconds"),
        case_ids=_case_order(order),
        thresholds=thresholds,
        environment=_environment(cfg),
        baseline=_system(flat, "baseline", thresholds),
        pipeline=_system(flat, "pipeline", thresholds),
        rows=flat,
        warnings=warnings,
    )


def format_summary(summary: EvalSummary) -> str:
    """The markdown side-by-side report. Ends with a newline."""
    titles = {
        "full": "Full eval run",
        "smoke": "Smoke eval run",
        "slice": "Eval slice",
    }
    env = summary.environment
    lines = [
        f"# {titles[summary.kind]}",
        "",
        _intro(summary),
        "",
        f"Eval time: {summary.eval_timestamp or 'unknown'}",
        f"Summarized: {summary.summarized_at}",
    ]
    if summary.eval_id:
        lines.append(f"Eval id: {summary.eval_id}")
    lines += [
        "",
        "## Models",
        "",
        "| Tier | Model |",
        "| --- | --- |",
        f"| Strong | {env.strong_model} |",
        f"| Fast | {env.fast_model} |",
        f"| Judge | {env.judge_model} |",
        "",
        (
            f"Retrieval: {'on' if env.retrieval_enabled else 'off'}. "
            f"Budgets: {env.max_tokens_per_run} tokens, {env.max_wall_clock_seconds}s, "
            f"{env.max_revisions} revisions. "
            f"Retries: {env.max_llm_retries}, waiting "
            f"{env.retry_initial_wait_seconds:g}s to {env.retry_max_wait_seconds:g}s. "
            f"Prompts: baseline {env.baseline_prompt}, planner {env.planner_prompt}, "
            f"writer {env.writer_prompt}, critic {env.critic_prompt}, "
            f"reviser {env.reviser_prompt}, judge {env.judge_prompt}."
        ),
        "",
        "The models, budgets and prompts are the settings of the process that wrote this summary.",
        "",
        "## Side by side",
        "",
        _comparison_table(summary),
        "",
        "## Per brief",
        "",
        _brief_table(summary),
    ]
    if summary.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in summary.warnings]
    lines += [
        "",
        "CI runs the five-case smoke eval and does not compare scores to this file.",
        "",
    ]
    return "\n".join(lines)


def write_summary(summary: EvalSummary, stem: Path) -> tuple[Path, Path]:
    """Write ``<stem>.md`` and ``<stem>.json``. A ``.md`` or ``.json`` suffix is stripped."""
    path = stem
    if path.suffix.lower() in {".md", ".json"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = path.with_suffix(".md")
    json_path = path.with_suffix(".json")
    markdown_path.write_text(format_summary(summary), encoding="utf-8")
    json_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return markdown_path, json_path


def _intro(summary: EvalSummary) -> str:
    count = len(summary.case_ids)
    briefs = f"{count} brief" if count == 1 else f"{count} briefs"
    overall = summary.thresholds.min_overall
    floor = summary.thresholds.min_per_criterion
    return (
        f"{briefs.capitalize()} through the baseline and the pipeline. "
        "Each brief counts once. Judge scores are the mean of the per-criterion means, "
        "on the 1-5 scale. A graded brief passes when that mean is at least "
        f"{overall:g} and every criterion mean is at least {floor}. "
        "Deterministic checks are the six hard checks. "
        "The baseline has no critic, so its flagged rate is not scored."
    )


def _comparison_table(summary: EvalSummary) -> str:
    base = summary.baseline
    pipe = summary.pipeline
    names = _criterion_names(base, pipe)
    rows = [
        ("Briefs", str(base.rows), str(pipe.rows)),
        ("Provider errors", str(base.errors), str(pipe.errors)),
        ("Partial or failed runs", str(base.incomplete), str(pipe.incomplete)),
        (
            "Deterministic pass rate",
            _pct(base.deterministic_passes, base.rows),
            _pct(pipe.deterministic_passes, pipe.rows),
        ),
        ("Judge mean", _num(base.judge_mean), _num(pipe.judge_mean)),
    ]
    rows += [
        (name, _num(base.judge_criteria.get(name)), _num(pipe.judge_criteria.get(name)))
        for name in names
    ]
    rows += [
        (
            "Judge pass rate",
            _graded(base.judge_passes, base.judge_compared),
            _graded(pipe.judge_passes, pipe.judge_compared),
        ),
        ("Judge min / median / max", _spread(base), _spread(pipe)),
        ("Scores 1-2, 2-3, 3-4, 4-5", _bands(base), _bands(pipe)),
        ("Flagged-variant rate", _flagged(base), _flagged(pipe)),
        ("Mean tokens per brief", _whole(base.mean_tokens), _whole(pipe.mean_tokens)),
        ("Mean cost per brief", _usd(base.mean_cost_usd), _usd(pipe.mean_cost_usd)),
        ("Total cost", _usd(base.total_cost_usd), _usd(pipe.total_cost_usd)),
        ("Total tokens", str(base.total_tokens), str(pipe.total_tokens)),
        (
            "Mean latency per brief",
            _seconds(base.mean_latency_ms),
            _seconds(pipe.mean_latency_ms),
        ),
    ]
    if base.crosscheck_mean is not None or pipe.crosscheck_mean is not None:
        rows.append(("Cross-check mean", _num(base.crosscheck_mean), _num(pipe.crosscheck_mean)))
    header = "| | Baseline | Pipeline |"
    rule = "| --- | --- | --- |"
    body = [f"| {label} | {left} | {right} |" for label, left, right in rows]
    return "\n".join([header, rule, *body])


def _brief_table(summary: EvalSummary) -> str:
    by_case: dict[str, dict[str, RowSummary]] = defaultdict(dict)
    for row in summary.rows:
        by_case[row.case_id][row.system] = row
    header = (
        "| Case | Baseline judge | Pipeline judge | Baseline checks | Pipeline checks "
        "| Baseline tokens | Pipeline tokens | Baseline cost | Pipeline cost "
        "| Baseline latency | Pipeline latency | Pipeline flagged |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    body = [_brief_line(case_id, by_case.get(case_id, {})) for case_id in summary.case_ids]
    return "\n".join([header, rule, *body])


def _brief_line(case_id: str, systems: dict[str, RowSummary]) -> str:
    baseline = systems.get("baseline")
    pipeline = systems.get("pipeline")
    cells = [
        case_id,
        _judge_cell(baseline),
        _judge_cell(pipeline),
        _check_cell(baseline),
        _check_cell(pipeline),
        _token_cell(baseline),
        _token_cell(pipeline),
        _cost_cell(baseline),
        _cost_cell(pipeline),
        _latency_cell(baseline),
        _latency_cell(pipeline),
        _flag_cell(pipeline),
    ]
    return "| " + " | ".join(cells) + " |"


def _system(
    rows: list[RowSummary], system: Literal["baseline", "pipeline"], thresholds: SummaryThresholds
) -> SystemSummary:
    chosen = [row for row in rows if row.system == system]
    scores = [row.judge_score for row in chosen if row.judge_score is not None]
    compared = [row for row in chosen if row.judge_score is not None and row.judge_criteria]
    passes = sum(1 for row in compared if _meets(row, thresholds))
    criteria = _mean_criteria(compared)
    tokens = [row.tokens for row in chosen if row.tokens is not None]
    costs = [row.cost_usd for row in chosen if row.cost_usd is not None]
    latencies = [row.latency_ms for row in chosen if row.latency_ms is not None]
    cross = [row.crosscheck_score for row in chosen if row.crosscheck_score is not None]
    flagged, variants = _flags(chosen)
    bands = dict.fromkeys(_BANDS, 0)
    for score in scores:
        bands[_band(score)] += 1
    return SystemSummary(
        rows=len(chosen),
        errors=sum(1 for row in chosen if row.error is not None),
        incomplete=sum(1 for row in chosen if row.status in {"partial", "failed"}),
        deterministic_passes=sum(1 for row in chosen if row.deterministic_pass),
        judge_scored=len(scores),
        judge_compared=len(compared),
        judge_passes=passes,
        judge_mean=_avg(scores),
        judge_criteria=criteria,
        judge_pass_rate=(passes / len(compared)) if compared else None,
        judge_min=min(scores) if scores else None,
        judge_median=_median(scores),
        judge_max=max(scores) if scores else None,
        judge_bands=bands,
        crosscheck_mean=_avg(cross),
        flagged_variants=flagged,
        variants=variants,
        flagged_rate=(flagged / variants) if flagged is not None and variants else None,
        total_tokens=sum(tokens),
        total_cost_usd=sum(costs),
        mean_tokens=_avg_num(tokens),
        mean_cost_usd=_avg(costs),
        mean_latency_ms=_avg(latencies),
    )


def _flags(rows: list[RowSummary]) -> tuple[int | None, int | None]:
    flagged = 0
    variants = 0
    scored = False
    for row in rows:
        if row.flagged_variants is None:
            continue
        scored = True
        flagged += row.flagged_variants
        variants += row.variant_count
    if not scored or variants == 0:
        return None, None
    return flagged, variants


def _meets(row: RowSummary, thresholds: SummaryThresholds) -> bool:
    score = row.judge_score
    if score is None or not row.judge_criteria:
        return False
    if score < thresholds.min_overall:
        return False
    return all(value >= thresholds.min_per_criterion for value in row.judge_criteria.values())


def _mean_criteria(rows: list[RowSummary]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for name, value in row.judge_criteria.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def _row(raw: dict[str, Any], index: int) -> RowSummary:
    parsed = _parse_output(raw)
    system = _system_name(raw, parsed)
    case_id = _case_id(raw, parsed)
    if system is None or case_id is None:
        raise ResultsError(f"result row {index} has no case id or system")
    checks = _checks(raw)
    judge_score, criteria, judge_error = _judge(raw)
    crosscheck, _, _ = _grade(raw, "judge_crosscheck")
    flagged, variants = _variant_flags(parsed)
    brand_id = parsed.brand_id if parsed is not None else _brand_id(case_id)
    if parsed is None:
        status = "error"
        error = _provider_error(raw)
        tokens = None
        cost = None
    else:
        status = parsed.status
        error = None
        tokens = parsed.usage.total_tokens
        cost = parsed.usage.cost_usd
    return RowSummary(
        case_id=case_id,
        system=system,
        brand_id=brand_id,
        status=status,
        deterministic_pass=all(checks.values()),
        deterministic=checks,
        judge_score=judge_score,
        judge_criteria=criteria,
        judge_error=judge_error,
        crosscheck_score=crosscheck,
        flagged_variants=flagged,
        variant_count=variants if parsed is not None else 0,
        tokens=tokens,
        cost_usd=cost,
        latency_ms=_latency(raw),
        error=error,
    )


def _variant_flags(parsed: EvalOutput | None) -> tuple[int | None, int]:
    if parsed is None:
        return None, 0
    decided = [item.flagged for item in parsed.variants if item.flagged is not None]
    if not decided:
        return None, len(parsed.variants)
    return sum(1 for flag in decided if flag), len(decided)


def _checks(raw: dict[str, Any]) -> dict[str, bool]:
    found, named = _metric_passes(raw)
    return {
        metric: found[metric] if metric in found else _named_pass(named, metric)
        for metric in DETERMINISTIC_METRICS
    }


def _judge(raw: dict[str, Any]) -> tuple[float | None, dict[str, float], str | None]:
    score, criteria, error = _grade(raw, "judge")
    # A component that refused to grade has an error string. namedScores is only
    # a fallback when promptfoo recorded the metric without a component result.
    if error is not None or score is not None or criteria:
        return score, criteria, error
    value = _named_scores(raw).get("judge")
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value), {}, None
    return None, {}, None


def _grade(raw: dict[str, Any], metric: str) -> tuple[float | None, dict[str, float], str | None]:
    for component in _components(raw):
        if _metric(component) != metric:
            continue
        reason = component.get("reason")
        text = reason if isinstance(reason, str) else ""
        message = " ".join(text.split())
        if component.get("pass") is not True:
            return None, {}, message or f"{metric} did not grade the row"
        value = component.get("score")
        score = (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )
        criteria = _criteria(component, text)
        # promptfoo records a weight-0 rubric as pass=true even when the grader
        # refused. A finished grade is on the 1-5 scale, so a score of 0 is not a mean.
        if (score is None or score <= 0) and criteria:
            score = sum(criteria.values()) / len(criteria)
        if score is None or score <= 0:
            return None, {}, message or f"{metric} did not grade the row"
        return score, criteria, None
    return None, {}, None


def _criteria(component: dict[str, Any], reason: str) -> dict[str, float]:
    metadata = component.get("metadata")
    if isinstance(metadata, dict):
        raw = metadata.get("criteria")
        parsed = _number_map(raw)
        if parsed:
            return parsed
    return _criteria_from_reason(reason)


def _criteria_from_reason(reason: str) -> dict[str, float]:
    first = reason.splitlines()[0] if reason else ""
    match = _MEAN_LINE.match(first.strip())
    if match is None:
        return {}
    return {name: float(value) for name, value in _CRITERION.findall(match.group(1))}


def _number_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, int | float) and not isinstance(value, bool):
            parsed[key] = float(value)
    return parsed


def _metric_passes(raw: dict[str, Any]) -> tuple[dict[str, bool], dict[str, float]]:
    found: dict[str, bool] = {}
    for component in _components(raw):
        metric = _metric(component)
        if metric is not None:
            found[metric] = component.get("pass") is True
    return found, _named_scores(raw)


def _named_pass(named: dict[str, float], metric: str) -> bool:
    return named.get(metric) == 1


def _named_scores(raw: dict[str, Any]) -> dict[str, float]:
    scores = raw.get("namedScores")
    return _number_map(scores)


def _components(raw: dict[str, Any]) -> list[dict[str, Any]]:
    grading = raw.get("gradingResult")
    if not isinstance(grading, dict):
        return []
    components = grading.get("componentResults")
    if not isinstance(components, list):
        return []
    return [item for item in components if isinstance(item, dict)]


def _metric(component: dict[str, Any]) -> str | None:
    assertion = component.get("assertion")
    if not isinstance(assertion, dict):
        return None
    metric = assertion.get("metric")
    if isinstance(metric, str) and metric.strip():
        return metric.strip()
    if assertion.get("type") == "is-json":
        return "valid_json"
    return None


def _parse_output(raw: dict[str, Any]) -> EvalOutput | None:
    response = raw.get("response")
    if not isinstance(response, dict):
        return None
    output = response.get("output")
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        try:
            payload: object = json.loads(text)
        except json.JSONDecodeError:
            return None
    elif isinstance(output, dict):
        payload = output
    else:
        return None
    try:
        return EvalOutput.model_validate(payload)
    except ValidationError:
        return None


def _system_name(
    raw: dict[str, Any], parsed: EvalOutput | None
) -> Literal["baseline", "pipeline"] | None:
    provider = raw.get("provider")
    if isinstance(provider, dict):
        label = provider.get("label")
        if label == "baseline":
            return "baseline"
        if label == "pipeline":
            return "pipeline"
    if parsed is not None:
        return parsed.system
    if isinstance(provider, dict):
        provider_id = provider.get("id")
        if isinstance(provider_id, str):
            if "baseline" in provider_id:
                return "baseline"
            if "pipeline" in provider_id:
                return "pipeline"
    return None


def _case_id(raw: dict[str, Any], parsed: EvalOutput | None) -> str | None:
    variables = raw.get("vars")
    if isinstance(variables, dict):
        value = variables.get("case_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    description = raw.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    prompt = raw.get("prompt")
    if isinstance(prompt, dict):
        raw_prompt = prompt.get("raw")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            return raw_prompt.strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    if parsed is not None:
        return parsed.case_id
    return None


def _provider_error(raw: dict[str, Any]) -> str:
    value = raw.get("error")
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())
    return "no eval row"


def _latency(raw: dict[str, Any]) -> float | None:
    value = raw.get("latencyMs")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value < 0:
        return None
    return float(value)


def _brand_id(case_id: str) -> str | None:
    try:
        return load_case(case_id).brand_id
    except CaseLoadError:
        return None


def _export_rows(payload: object) -> tuple[list[dict[str, Any]], str | None, str | None]:
    if isinstance(payload, list):
        return _dicts(payload), None, None
    if not isinstance(payload, dict):
        raise ResultsError("promptfoo export must be a JSON object or a list of rows")
    eval_id = _text(payload.get("evalId"))
    results = payload.get("results")
    if isinstance(results, list):
        return _dicts(results), eval_id, _text(payload.get("timestamp"))
    if isinstance(results, dict):
        inner = results.get("results")
        if isinstance(inner, list):
            nested_id = _text(results.get("evalId")) or eval_id
            return _dicts(inner), nested_id, _text(results.get("timestamp"))
    raise ResultsError("promptfoo export has no results list")


def _dicts(rows: list[object]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ResultsError(f"result row {index} is not an object")
        parsed.append(row)
    return parsed


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _environment(settings: Settings) -> SummaryEnvironment:
    return SummaryEnvironment(
        strong_model=settings.models.strong,
        fast_model=settings.models.fast,
        judge_model=settings.models.judge,
        retrieval_enabled=settings.retrieval_enabled,
        max_tokens_per_run=settings.budgets.max_tokens_per_run,
        max_wall_clock_seconds=settings.budgets.max_wall_clock_seconds,
        max_revisions=settings.budgets.max_revisions,
        max_llm_retries=settings.budgets.max_llm_retries,
        retry_initial_wait_seconds=settings.budgets.retry_initial_wait_seconds,
        retry_max_wait_seconds=settings.budgets.retry_max_wait_seconds,
        baseline_prompt=settings.baseline_prompt_version,
        planner_prompt=settings.planner_prompt_version,
        writer_prompt=settings.writer_prompt_version,
        critic_prompt=settings.critic_prompt_version,
        reviser_prompt=settings.reviser_prompt_version,
        judge_prompt=settings.judge_prompt_version,
    )


def _kind(case_ids: list[str]) -> SummaryKind:
    found = set(case_ids)
    if found == set(list_case_ids()) and len(found) == len(list_case_ids()):
        return "full"
    if found == set(SMOKE_CASE_IDS) and len(found) == len(SMOKE_CASE_IDS):
        return "smoke"
    return "slice"


def _case_order(seen: list[str]) -> list[str]:
    known = list_case_ids()
    rank = {case_id: index for index, case_id in enumerate(known)}
    return sorted(seen, key=lambda case_id: (rank.get(case_id, len(known)), case_id))


def _ordered(parsed: dict[tuple[str, str], RowSummary], case_id: str) -> list[RowSummary]:
    return [parsed[(system, case_id)] for system in _SYSTEMS if (system, case_id) in parsed]


def _criterion_names(baseline: SystemSummary, pipeline: SystemSummary) -> list[str]:
    names: list[str] = []
    for source in (baseline.judge_criteria, pipeline.judge_criteria):
        for name in source:
            if name not in names:
                names.append(name)
    return names


def _band(score: float) -> str:
    if score < 2:
        return "1-2"
    if score < 3:
        return "2-3"
    if score < 4:
        return "3-4"
    return "4-5"


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _avg_num(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _pct(numer: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{100.0 * numer / denom:.1f}% ({numer}/{denom})"


def _graded(passes: int, compared: int) -> str:
    if compared == 0:
        return "—"
    return f"{100.0 * passes / compared:.1f}% ({passes}/{compared} graded)"


def _spread(system: SystemSummary) -> str:
    if system.judge_min is None or system.judge_median is None or system.judge_max is None:
        return "—"
    return f"{system.judge_min:.2f} / {system.judge_median:.2f} / {system.judge_max:.2f}"


def _bands(system: SystemSummary) -> str:
    if system.judge_scored == 0:
        return "—"
    return ", ".join(str(system.judge_bands[band]) for band in _BANDS)


def _flagged(system: SystemSummary) -> str:
    if system.flagged_variants is None or system.variants is None or system.flagged_rate is None:
        return "not scored"
    rate = 100.0 * system.flagged_rate
    return f"{rate:.1f}% ({system.flagged_variants}/{system.variants})"


def _num(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _whole(value: float | None) -> str:
    if value is None:
        return "—"
    return str(round(value))


def _usd(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.6f}"


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:.1f}s"


def _judge_cell(row: RowSummary | None) -> str:
    if row is None:
        return "—"
    if row.error is not None:
        return "error"
    if row.judge_score is None:
        return "ungraded"
    return f"{row.judge_score:.2f}"


def _check_cell(row: RowSummary | None) -> str:
    if row is None:
        return "—"
    if row.error is not None:
        return "error"
    return "pass" if row.deterministic_pass else "fail"


def _token_cell(row: RowSummary | None) -> str:
    if row is None or row.tokens is None:
        return "—"
    return str(row.tokens)


def _cost_cell(row: RowSummary | None) -> str:
    if row is None:
        return "—"
    return _usd(row.cost_usd)


def _latency_cell(row: RowSummary | None) -> str:
    if row is None:
        return "—"
    return _seconds(row.latency_ms)


def _flag_cell(row: RowSummary | None) -> str:
    if row is None or row.flagged_variants is None:
        return "—"
    return f"{row.flagged_variants}/{row.variant_count}"
