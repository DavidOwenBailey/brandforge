"""Judge calibration (BF-37).

Fifteen hand-scored outputs live in ``evals/calibration/``. ``calibrate`` sends
each one to the judge and compares the 1-5 criterion scores. Agreement is
quadratic weighted kappa on that fixed scale. Kappa below 0.60 means the
rubric anchors should be tuned. The rules are in ADR 0030.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from brandforge.brands import BrandLoadError, load_brand
from brandforge.config import Settings, get_settings
from brandforge.evals.cases import CASES_DIR, CaseLoadError, load_case
from brandforge.evals.judge import StructuredCompleter, score_row
from brandforge.evals.promptfoo import EvalOutput, EvalUsage, EvalVariant
from brandforge.models import Channel, NonEmptyStr, Score, Usage

# Landis and Koch call 0.60 the start of "substantial" agreement. Below that,
# the architecture says to tune the rubric rather than trust the judge mean.
KAPPA_LOW = 0.6
# A one-point gap is the "within one" band. The report lists wider gaps.
LARGE_GAP = 2
_SCALE = (1, 2, 3, 4, 5)


class CalibrationError(Exception):
    """A calibration file is missing, malformed, or does not match its case."""


class _File(BaseModel):
    """One YAML file: the copy, and the hand score the judge must not see."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    case_id: NonEmptyStr
    channel: Channel
    headline: NonEmptyStr
    body: NonEmptyStr
    cta: NonEmptyStr
    scores: dict[str, Score]
    note: NonEmptyStr

    @field_validator("scores")
    @classmethod
    def _scores_are_plain_ints(cls, scores: dict[str, int]) -> dict[str, int]:
        for name, score in scores.items():
            if isinstance(score, bool):
                raise ValueError(f"{name} must be an integer from 1 to 5")
        return scores


@dataclass(frozen=True)
class CalibrationItem:
    """One authored output and the scores a person gave it.

    ``row_json`` is the eval row the judge scores. The hand scores and the note
    stay here, so they are not part of that row.
    """

    id: str
    case_id: str
    brand_id: str
    brand_version: str
    rubric_version: str
    channel: Channel
    headline: str
    body: str
    cta: str
    scores: dict[str, int]
    note: str

    def row_json(self) -> str:
        """The baseline-shaped row ``score_row`` already accepts.

        These files are authored fixtures, not a live baseline run. The baseline
        shape is the one with no critic flag. The judge does not read ``system``.
        """
        row = EvalOutput(
            system="baseline",
            case_id=self.case_id,
            brand_id=self.brand_id,
            brand_version=self.brand_version,
            rubric_version=self.rubric_version,
            status="complete",
            revision_count=0,
            variants=[
                EvalVariant(
                    id="v1",
                    channel=self.channel,
                    headline=self.headline,
                    body=self.body,
                    cta=self.cta,
                    flagged=None,
                )
            ],
            errors=[],
            usage=EvalUsage(
                input_tokens=0,
                output_tokens=0,
                cache_write_tokens=0,
                cache_read_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
            ),
        )
        return row.model_dump_json()


@dataclass(frozen=True)
class ScorePair:
    """One criterion on one output, scored by a person and by the judge."""

    item_id: str
    brand_id: str
    criterion: str
    human: int
    judge: int


@dataclass(frozen=True)
class Agreement:
    """Agreement on a set of criterion pairs. ``kappa`` is undefined when the
    scores do not spread, because chance agreement is then the whole scale."""

    count: int
    kappa: float | None
    exact: int
    within_one: int
    mae: float
    mean_signed_error: float


@dataclass(frozen=True)
class CalibrationReport:
    """What ``brandforge calibrate`` prints.

    ``overall`` counts every criterion score. ``row_mean_mae`` is the same
    comparison rolled up to the 1-5 mean the eval table shows. ``low`` is
    ``None`` when kappa is undefined.
    """

    outputs: int
    scored: int
    unscored: tuple[tuple[str, str], ...]
    overall: Agreement
    by_criterion: dict[str, Agreement]
    by_brand: dict[str, Agreement]
    disagreements: tuple[ScorePair, ...]
    row_mean_mae: float
    row_mean_signed_error: float
    model: str
    prompt_version: str
    usage: Usage

    @property
    def low(self) -> bool | None:
        if self.overall.kappa is None:
            return None
        return self.overall.kappa < KAPPA_LOW

    @property
    def verdict(self) -> str:
        if self.scored == 0:
            return "Verdict: there is no agreement figure, because the judge scored nothing."
        kappa = self.overall.kappa
        if kappa is None:
            return (
                "Verdict: kappa is undefined. The scores sit on too little of the "
                "1-5 scale to measure agreement."
            )
        if kappa < KAPPA_LOW:
            return (
                f"Verdict: agreement is low (kappa {kappa:.2f}, below {KAPPA_LOW:.2f}). "
                "Tune the rubric anchors before treating the judge mean as your score."
            )
        return (
            f"Verdict: agreement is {kappa:.2f}, at or above {KAPPA_LOW:.2f}. The rubric can stand."
        )


def calibration_dir() -> Path:
    """``evals/calibration``, beside the case files."""
    return CASES_DIR.parent / "calibration"


def quadratic_weighted_kappa(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Quadratic weighted Cohen's kappa on the fixed 1-5 scale.

    Weights are ``(human - judge) ** 2 / 16``, so a gap of 4 counts as total
    disagreement and a gap of 1 counts as one sixteenth of that. Levels nobody
    used stay in the scale. The result is ``None`` when there are no pairs, a
    score falls outside 1-5, or every score landed on one level (expected
    disagreement is then zero, and kappa would be 0/0).
    """
    if not pairs:
        return None
    index = {score: i for i, score in enumerate(_SCALE)}
    for human, judge in pairs:
        if human not in index or judge not in index:
            raise ValueError(f"scores must be 1 to 5, got {human} and {judge}")

    size = len(_SCALE)
    observed = [[0.0] * size for _ in range(size)]
    for human, judge in pairs:
        observed[index[human]][index[judge]] += 1.0
    count = float(len(pairs))
    row_sum = [sum(row) for row in observed]
    col_sum = [sum(observed[row][col] for row in range(size)) for col in range(size)]
    expected = [[row_sum[row] * col_sum[col] / count for col in range(size)] for row in range(size)]

    def weight(row: int, col: int) -> float:
        return (row - col) ** 2 / (size - 1) ** 2

    observed_disagreement = sum(
        weight(row, col) * observed[row][col] for row in range(size) for col in range(size)
    )
    expected_disagreement = sum(
        weight(row, col) * expected[row][col] for row in range(size) for col in range(size)
    )
    if expected_disagreement == 0.0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def load_calibration(directory: Path | None = None) -> list[CalibrationItem]:
    """Load every ``*.yaml`` file in the calibration directory, in filename order.

    Raises `CalibrationError` when the directory is missing or empty, or when
    any file fails validation. Loading does not call a model.
    """
    folder = calibration_dir() if directory is None else directory
    if not folder.is_dir():
        raise CalibrationError(f"calibration directory not found: {folder}")
    paths = sorted(folder.glob("*.yaml"))
    if not paths:
        raise CalibrationError(f"no calibration outputs in {folder}")
    return [_load_file(path) for path in paths]


def calibrate(
    items: Sequence[CalibrationItem],
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter | None = None,
) -> CalibrationReport:
    """Score each output with the judge and compare the criterion integers.

    One judge call per output, through ``score_row``. A row the judge cannot
    score is listed and left out of the kappa. Pass ``complete`` to use a fake
    gateway. The default calls the judge tier.
    """
    if not items:
        raise CalibrationError("no calibration outputs")
    cfg = settings if settings is not None else get_settings()
    pairs: list[ScorePair] = []
    unscored: list[tuple[str, str]] = []
    human_means: list[float] = []
    judge_means: list[float] = []
    usage = Usage()
    model = cfg.models.judge
    prompt_version = cfg.judge_prompt_version

    for item in items:
        grade = score_row(item.row_json(), settings=cfg, complete=complete)
        usage = usage + grade.usage
        model = grade.model
        prompt_version = grade.prompt_version
        if not grade.passed:
            unscored.append((item.id, grade.reason))
            continue
        judged = grade.variants[0].scores
        for name, human in item.scores.items():
            pairs.append(ScorePair(item.id, item.brand_id, name, human, judged[name]))
        human_means.append(sum(item.scores.values()) / len(item.scores))
        judge_means.append(grade.score)

    by_criterion: dict[str, list[ScorePair]] = defaultdict(list)
    by_brand: dict[str, list[ScorePair]] = defaultdict(list)
    for pair in pairs:
        by_criterion[pair.criterion].append(pair)
        by_brand[pair.brand_id].append(pair)
    criterion_order = list(dict.fromkeys(pair.criterion for pair in pairs))
    gaps = tuple(pair for pair in pairs if abs(pair.human - pair.judge) >= LARGE_GAP)
    return CalibrationReport(
        outputs=len(items),
        scored=len(items) - len(unscored),
        unscored=tuple(unscored),
        overall=_agreement(pairs),
        by_criterion={name: _agreement(by_criterion[name]) for name in criterion_order},
        by_brand={name: _agreement(by_brand[name]) for name in sorted(by_brand)},
        disagreements=gaps,
        row_mean_mae=_mean_abs(human_means, judge_means),
        row_mean_signed_error=_mean_signed(judge_means, human_means),
        model=model,
        prompt_version=prompt_version,
        usage=usage,
    )


def run_calibration(
    *,
    directory: Path | None = None,
    settings: Settings | None = None,
    complete: StructuredCompleter | None = None,
) -> CalibrationReport:
    """Load the sample and compare it with the judge."""
    return calibrate(load_calibration(directory), settings=settings, complete=complete)


def format_report(report: CalibrationReport) -> str:
    """The text ``brandforge calibrate`` prints."""
    lines = [
        "Judge calibration",
        (
            f"{report.outputs} authored outputs, {report.scored} scored, "
            f"{report.overall.count} criterion scores."
        ),
        f"Judge: {report.model} (prompt {report.prompt_version}).",
        report.verdict,
    ]
    if report.usage.total_tokens or report.usage.cost_usd:
        lines.append(
            f"Judge usage: {report.usage.total_tokens} tokens, ${report.usage.cost_usd:.4f}."
        )
    if report.overall.count:
        stats = report.overall
        lines.extend(
            [
                "",
                f"Quadratic weighted kappa: {_kappa_text(stats.kappa)}",
                f"Exact agreement: {stats.exact}/{stats.count}",
                f"Within one point: {stats.within_one}/{stats.count}",
                f"Mean absolute error: {stats.mae:.2f}",
                f"Mean signed error (judge minus human): {stats.mean_signed_error:.2f}",
                f"Row-mean absolute error: {report.row_mean_mae:.2f}",
                f"Row-mean signed error (judge minus human): {report.row_mean_signed_error:.2f}",
                "",
                "By criterion",
                _table(
                    ["Criterion", "Kappa", "Exact", "Within one", "MAE", "Judge-human", "n"],
                    [_agreement_row(name, item) for name, item in report.by_criterion.items()],
                ),
                "",
                "By brand",
                _table(
                    ["Brand", "Kappa", "Exact", "Within one", "MAE", "Judge-human", "n"],
                    [_agreement_row(name, item) for name, item in report.by_brand.items()],
                ),
                "",
                "Gaps of two points or more",
            ]
        )
        if report.disagreements:
            lines.extend(
                f"{gap.item_id}  {gap.criterion}  human {gap.human}  judge {gap.judge}"
                for gap in report.disagreements
            )
        else:
            lines.append("(none)")
    if report.unscored:
        lines.extend(["", "Unscored"])
        lines.extend(f"{item_id}  {reason}" for item_id, reason in report.unscored)
    return "\n".join(lines)


def _load_file(path: Path) -> CalibrationItem:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CalibrationError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise CalibrationError(f"{path.name}: a calibration output must be a YAML mapping")
    try:
        parsed = _File.model_validate(raw)
    except ValidationError as exc:
        raise CalibrationError(f"{path.name}: {exc}") from exc
    if parsed.id != path.stem:
        raise CalibrationError(f"{path.name}: id {parsed.id!r} must match the filename")

    try:
        case = load_case(parsed.case_id)
    except CaseLoadError as exc:
        raise CalibrationError(f"{path.name}: {exc}") from exc
    if not parsed.id.startswith(f"{case.brand_id}_"):
        raise CalibrationError(f"{path.name}: id must start with the brand id {case.brand_id!r}")
    if parsed.channel not in case.brief.channels:
        requested = ", ".join(case.brief.channels)
        raise CalibrationError(
            f"{path.name}: channel {parsed.channel!r} is not in this brief ({requested})"
        )
    try:
        brand = load_brand(case.brand_id)
    except BrandLoadError as exc:
        raise CalibrationError(f"{path.name}: {exc}") from exc

    expected = [criterion.name for criterion in brand.rubric.criteria]
    if set(parsed.scores) != set(expected):
        raise CalibrationError(f"{path.name}: scores must be {', '.join(expected)}")
    return CalibrationItem(
        id=parsed.id,
        case_id=parsed.case_id,
        brand_id=case.brand_id,
        brand_version=brand.version,
        rubric_version=brand.rubric.version,
        channel=parsed.channel,
        headline=parsed.headline,
        body=parsed.body,
        cta=parsed.cta,
        scores={name: parsed.scores[name] for name in expected},
        note=parsed.note,
    )


def _agreement(pairs: Sequence[ScorePair]) -> Agreement:
    if not pairs:
        return Agreement(0, None, 0, 0, 0.0, 0.0)
    absolute = [abs(pair.human - pair.judge) for pair in pairs]
    signed = [pair.judge - pair.human for pair in pairs]
    count = len(pairs)
    return Agreement(
        count=count,
        kappa=quadratic_weighted_kappa([(pair.human, pair.judge) for pair in pairs]),
        exact=sum(1 for gap in absolute if gap == 0),
        within_one=sum(1 for gap in absolute if gap <= 1),
        mae=sum(absolute) / count,
        mean_signed_error=sum(signed) / count,
    )


def _mean_abs(human: Sequence[float], judge: Sequence[float]) -> float:
    if not human:
        return 0.0
    return sum(abs(left - right) for left, right in zip(human, judge, strict=True)) / len(human)


def _mean_signed(judge: Sequence[float], human: Sequence[float]) -> float:
    if not human:
        return 0.0
    return sum(left - right for left, right in zip(judge, human, strict=True)) / len(human)


def _agreement_row(label: str, stats: Agreement) -> list[str]:
    return [
        label,
        _kappa_text(stats.kappa),
        f"{stats.exact}/{stats.count}",
        f"{stats.within_one}/{stats.count}",
        f"{stats.mae:.2f}",
        f"{stats.mean_signed_error:.2f}",
        str(stats.count),
    ]


def _kappa_text(kappa: float | None) -> str:
    return "undefined" if kappa is None else f"{kappa:.2f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return "  ".join(padded).rstrip()

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(header), rule, *(line(row) for row in rows)])
