"""Eval summaries and the CI smoke set (BF-38). These tests make no model call."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic_settings import SettingsConfigDict
from typer.testing import CliRunner

from brandforge.config import Settings
from brandforge.evals.cases import CASES_DIR, list_case_ids, load_case
from brandforge.evals.results import (
    DETERMINISTIC_METRICS,
    SMOKE_CASE_IDS,
    EvalSummary,
    ResultsError,
    format_summary,
    summarize_export,
    summarize_path,
    write_summary,
)
from brandforge.interfaces import cli

runner = CliRunner()

BRIGHTLEAF = "brightleaf_01_spring_blossom"
LEDGERLY = "ledgerly_01_vat_reminders"
_CRITERIA = ("voice", "clarity", "call_to_action")


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def _variant(variant_id: str, channel: str, flagged: bool | None) -> dict[str, object]:
    return {
        "id": variant_id,
        "channel": channel,
        "headline": "Hello",
        "body": "A short line.",
        "cta": "Shop",
        "flagged": flagged,
    }


def _output(
    system: str,
    case_id: str,
    brand_id: str,
    variants: list[dict[str, object]],
    *,
    total_tokens: int,
    cost: float,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "system": system,
        "case_id": case_id,
        "brand_id": brand_id,
        "brand_version": "1.0",
        "rubric_version": "1.0",
        "status": status,
        "run_id": None,
        "trace_id": None,
        "revision_count": 0,
        "variants": variants,
        "errors": [],
        "usage": {
            "input_tokens": total_tokens,
            "output_tokens": 0,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": total_tokens,
            "cost_usd": cost,
        },
    }


def _check(metric: str, passed: bool = True) -> dict[str, object]:
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "ok" if passed else f"{metric} failed",
        "assertion": {"type": "python" if metric != "valid_json" else "is-json", "metric": metric},
    }


def _checks(**failed: bool) -> list[dict[str, object]]:
    return [_check(metric, metric not in failed) for metric in DETERMINISTIC_METRICS]


def _judge(
    score: float,
    criteria: dict[str, float] | None = None,
    *,
    passed: bool = True,
    metric: str = "judge",
    reason: str | None = None,
    metadata: bool = True,
) -> dict[str, object]:
    grades = criteria
    if grades is None:
        grades = dict.fromkeys(_CRITERIA, score)
    if reason is None:
        parts = ", ".join(f"{name} {value:.2f}" for name, value in grades.items())
        reason = f"mean {score:.2f} ({parts})"
    component: dict[str, object] = {
        "pass": passed,
        "score": score if passed else 0,
        "reason": reason if passed else "judge failed",
        "assertion": {"type": "llm-rubric", "metric": metric},
    }
    if passed and metadata:
        component["metadata"] = {"criteria": grades}
    return component


def _result(
    system: str,
    case_id: str,
    output: dict[str, object] | None,
    components: list[dict[str, object]] | None,
    *,
    latency: float = 1000,
    error: str | None = None,
    named: dict[str, float] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "description": case_id,
        "provider": {"id": f"file://providers/{system}.py", "label": system},
        "vars": {"case_id": case_id},
        "success": error is None,
        "score": 1 if error is None else 0,
        "latencyMs": latency,
        "namedScores": named or {},
        "response": {"output": json.dumps(output) if output is not None else ""},
    }
    if error is not None:
        row["error"] = error
    if components is not None:
        row["gradingResult"] = {
            "pass": all(item["pass"] for item in components),
            "score": 1,
            "reason": "ok",
            "componentResults": components,
        }
    return row


def _wrapped(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "evalId": "eval-1",
        "results": {
            "version": 3,
            "timestamp": "2026-10-10T12:00:00.000Z",
            "results": rows,
        },
    }


def _pair() -> list[dict[str, object]]:
    """Two briefs. One judge grade fails, and one pipeline call returns no row."""
    baseline_a = _output(
        "baseline",
        BRIGHTLEAF,
        "brightleaf",
        [_variant("v1", "email", None), _variant("v2", "social", None)],
        total_tokens=100,
        cost=0.01,
    )
    pipeline_a = _output(
        "pipeline",
        BRIGHTLEAF,
        "brightleaf",
        [_variant("v1", "email", False), _variant("v2", "social", True)],
        total_tokens=200,
        cost=0.02,
        status="partial",
    )
    baseline_b = _output(
        "baseline",
        LEDGERLY,
        "ledgerly",
        [_variant("v1", "email", None)],
        total_tokens=50,
        cost=0.005,
    )
    grades = {"voice": 4.0, "clarity": 4.0, "call_to_action": 4.0}
    return [
        _result(
            "baseline",
            BRIGHTLEAF,
            baseline_a,
            [*_checks(), _judge(4.0, grades)],
            latency=1000,
        ),
        _result(
            "pipeline",
            BRIGHTLEAF,
            pipeline_a,
            [*_checks(), _judge(5.0)],
            latency=2000,
        ),
        _result(
            "baseline",
            LEDGERLY,
            baseline_b,
            [*_checks(), _judge(4.0, passed=False)],
            latency=500,
            named={"judge": 4.0},
        ),
        _result(
            "pipeline",
            LEDGERLY,
            None,
            None,
            latency=100,
            error="GatewayError: down",
        ),
    ]


def test_summary_reports_both_systems_side_by_side() -> None:
    settings = IsolatedSettings()
    summary = summarize_export(_wrapped(_pair()), settings=settings)

    assert summary.kind == "slice"
    assert summary.eval_id == "eval-1"
    assert summary.eval_timestamp == "2026-10-10T12:00:00.000Z"
    assert summary.case_ids == [BRIGHTLEAF, LEDGERLY]
    assert summary.environment.strong_model == settings.models.strong
    assert summary.environment.max_wall_clock_seconds == settings.budgets.max_wall_clock_seconds
    assert summary.environment.max_llm_retries == settings.budgets.max_llm_retries
    assert summary.thresholds.min_overall == settings.thresholds.min_overall
    assert summary.thresholds.min_per_criterion == settings.thresholds.min_per_criterion

    baseline = summary.baseline
    assert baseline.rows == 2
    assert baseline.errors == 0
    assert baseline.incomplete == 0
    assert baseline.deterministic_passes == 2
    assert baseline.judge_scored == 1
    assert baseline.judge_compared == 1
    assert baseline.judge_passes == 1
    assert baseline.judge_mean == pytest.approx(4.0)
    assert baseline.judge_criteria["voice"] == pytest.approx(4.0)
    assert baseline.judge_pass_rate == pytest.approx(1.0)
    assert baseline.flagged_rate is None
    assert baseline.total_tokens == 150
    assert baseline.mean_tokens == pytest.approx(75)
    assert baseline.total_cost_usd == pytest.approx(0.015)
    assert baseline.mean_latency_ms == pytest.approx(750)

    pipeline = summary.pipeline
    assert pipeline.rows == 2
    assert pipeline.errors == 1
    assert pipeline.incomplete == 1
    assert pipeline.deterministic_passes == 1
    assert pipeline.judge_mean == pytest.approx(5.0)
    assert pipeline.judge_passes == 1
    assert pipeline.flagged_variants == 1
    assert pipeline.variants == 2
    assert pipeline.flagged_rate == pytest.approx(0.5)
    assert pipeline.total_tokens == 200
    assert pipeline.mean_tokens == pytest.approx(200)
    assert pipeline.mean_latency_ms == pytest.approx(1050)
    failed = next(row for row in summary.rows if row.error is not None)
    assert failed.case_id == LEDGERLY
    assert failed.brand_id == "ledgerly"
    assert failed.deterministic_pass is False
    assert failed.tokens is None

    text = format_summary(summary)
    assert "not scored" in text
    assert "50.0% (1/2)" in text
    assert BRIGHTLEAF in text
    assert "ungraded" in text
    assert "error" in text


def test_a_weight_zero_pass_does_not_turn_a_refusal_into_zero() -> None:
    """promptfoo stores weight 0 as pass=true when the grader returns score 0."""
    refused = _judge(4.0, passed=False)
    refused["pass"] = True
    refused["score"] = 0
    refused["reason"] = "output has no variants"
    refused["metadata"] = {"criteria": {}}
    row = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=10,
            cost=0.001,
        ),
        [*_checks(), refused],
        named={"judge": 0},
    )
    summary = summarize_export(_wrapped([row]), settings=IsolatedSettings())
    assert summary.rows[0].judge_score is None
    assert summary.rows[0].judge_error == "output has no variants"
    assert summary.baseline.judge_scored == 0
    assert summary.baseline.judge_mean is None


def test_a_failed_judge_grade_is_not_a_score_of_zero() -> None:
    row = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=10,
            cost=0.001,
        ),
        [*_checks(), _judge(4.0, passed=False)],
        named={"judge": 4.0},
    )
    summary = summarize_export(_wrapped([row]), settings=IsolatedSettings())
    assert summary.baseline.judge_scored == 0
    assert summary.baseline.judge_mean is None
    assert summary.rows[0].judge_score is None
    assert summary.rows[0].judge_error == "judge failed"


def test_an_assertion_failure_keeps_the_row() -> None:
    row = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=40,
            cost=0.002,
        ),
        [*_checks(channel_limits=True), _judge(4.5)],
        error="headline is too long",
    )
    summary = summarize_export([row], settings=IsolatedSettings())
    parsed = summary.rows[0]
    assert parsed.error is None
    assert parsed.tokens == 40
    assert parsed.deterministic_pass is False
    assert parsed.deterministic["channel_limits"] is False
    assert parsed.deterministic["banned_words"] is True
    assert parsed.judge_score == pytest.approx(4.5)


def test_judge_pass_rate_uses_the_critic_thresholds() -> None:
    def row(score: float, voice: float) -> dict[str, object]:
        criteria = {"voice": voice, "clarity": 5.0, "call_to_action": 5.0}
        return _result(
            "pipeline",
            BRIGHTLEAF,
            _output(
                "pipeline",
                BRIGHTLEAF,
                "brightleaf",
                [_variant("v1", "email", False)],
                total_tokens=10,
                cost=0.001,
            ),
            [*_checks(), _judge(score, criteria)],
        )

    on_the_line = summarize_export([row(4.0, 3.0)], settings=IsolatedSettings()).pipeline
    assert on_the_line.judge_passes == 1

    low_criterion = summarize_export([row(4.0, 2.0)], settings=IsolatedSettings()).pipeline
    assert low_criterion.judge_passes == 0
    assert low_criterion.judge_mean == pytest.approx(4.0)

    low_overall = summarize_export([row(3.9, 4.0)], settings=IsolatedSettings()).pipeline
    assert low_overall.judge_passes == 0


def test_distribution_and_median() -> None:
    scores = (1.5, 2.5, 3.5, 5.0)
    rows = []
    for index, score in enumerate(scores):
        case_id = list_case_ids()[index]
        case = load_case(case_id)
        rows.append(
            _result(
                "baseline",
                case_id,
                _output(
                    "baseline",
                    case_id,
                    case.brand_id,
                    [_variant("v1", "email", None)],
                    total_tokens=10,
                    cost=0.001,
                ),
                [*_checks(), _judge(score)],
            )
        )
    baseline = summarize_export(rows, settings=IsolatedSettings()).baseline
    assert baseline.judge_min == pytest.approx(1.5)
    assert baseline.judge_median == pytest.approx(3.0)
    assert baseline.judge_max == pytest.approx(5.0)
    assert baseline.judge_bands == {"1-2": 1, "2-3": 1, "3-4": 1, "4-5": 1}


def test_reason_line_supplies_criteria_when_metadata_is_missing() -> None:
    reason = "mean 4.20 (voice 4.00, clarity 4.50, call_to_action 4.10)\nv1 email: voice 4"
    row = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=10,
            cost=0.001,
        ),
        [*_checks(), _judge(4.2, metadata=False, reason=reason)],
    )
    parsed = summarize_export([row], settings=IsolatedSettings()).rows[0]
    assert parsed.judge_score == pytest.approx(4.2)
    assert parsed.judge_criteria["clarity"] == pytest.approx(4.5)
    assert parsed.judge_criteria["call_to_action"] == pytest.approx(4.1)


def test_named_scores_stand_in_when_there_is_no_component() -> None:
    row = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=10,
            cost=0.001,
        ),
        None,
        named=dict.fromkeys((*DETERMINISTIC_METRICS, "judge"), 1) | {"judge": 3.5},
    )
    summary = summarize_export([row], settings=IsolatedSettings())
    assert summary.rows[0].deterministic_pass is True
    assert summary.rows[0].judge_score == pytest.approx(3.5)
    assert summary.baseline.judge_compared == 0
    assert summary.baseline.judge_pass_rate is None
    assert summary.baseline.judge_mean == pytest.approx(3.5)


def test_is_json_without_a_metric_still_counts() -> None:
    components: list[dict[str, object]] = [
        {
            "pass": True,
            "score": 1,
            "reason": "ok",
            "assertion": {"type": "is-json"},
        }
    ]
    named = {metric: 1.0 for metric in DETERMINISTIC_METRICS if metric != "valid_json"}
    row = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=10,
            cost=0.001,
        ),
        components,
        named=named,
    )
    assert summarize_export([row], settings=IsolatedSettings()).rows[0].deterministic_pass is True


def test_crosscheck_mean_stays_off_the_judge_mean() -> None:
    row = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=10,
            cost=0.001,
        ),
        [*_checks(), _judge(4.0), _judge(2.0, metric="judge_crosscheck")],
    )
    baseline = summarize_export([row], settings=IsolatedSettings()).baseline
    assert baseline.judge_mean == pytest.approx(4.0)
    assert baseline.crosscheck_mean == pytest.approx(2.0)


def test_a_later_duplicate_row_wins() -> None:
    first = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=10,
            cost=0.001,
        ),
        [*_checks(), _judge(2.0)],
    )
    second = _result(
        "baseline",
        BRIGHTLEAF,
        _output(
            "baseline",
            BRIGHTLEAF,
            "brightleaf",
            [_variant("v1", "email", None)],
            total_tokens=30,
            cost=0.003,
        ),
        [*_checks(), _judge(4.0)],
    )
    summary = summarize_export([first, second], settings=IsolatedSettings())
    assert summary.rows[0].judge_score == pytest.approx(4.0)
    assert summary.rows[0].tokens == 30
    assert len(summary.rows) == 1
    assert summary.warnings


def test_case_order_follows_the_dataset() -> None:
    rows = []
    for case_id in (LEDGERLY, BRIGHTLEAF):
        case = load_case(case_id)
        rows.append(
            _result(
                "pipeline",
                case_id,
                _output(
                    "pipeline",
                    case_id,
                    case.brand_id,
                    [_variant("v1", "email", False)],
                    total_tokens=10,
                    cost=0.001,
                ),
                _checks(),
            )
        )
    summary = summarize_export(rows, settings=IsolatedSettings())
    assert summary.case_ids == [BRIGHTLEAF, LEDGERLY]


def test_kind_matches_the_case_set() -> None:
    settings = IsolatedSettings()

    def errors(case_ids: tuple[str, ...] | list[str]) -> list[dict[str, object]]:
        return [
            _result("baseline", case_id, None, None, error="GatewayError: down")
            for case_id in case_ids
        ]

    assert summarize_export(errors(SMOKE_CASE_IDS), settings=settings).kind == "smoke"
    assert summarize_export(errors(list_case_ids()), settings=settings).kind == "full"
    assert summarize_export(errors((BRIGHTLEAF,)), settings=settings).kind == "slice"


def test_export_shapes_and_rejections(tmp_path: Path) -> None:
    row = _result("baseline", BRIGHTLEAF, None, None, error="GatewayError: down")
    assert summarize_export(_wrapped([row]), settings=IsolatedSettings()).eval_id == "eval-1"

    bare = tmp_path / "rows.json"
    bare.write_text(json.dumps([row]), encoding="utf-8")
    assert summarize_path(bare, settings=IsolatedSettings()).rows[0].case_id == BRIGHTLEAF

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ResultsError, match="not JSON"):
        summarize_path(broken, settings=IsolatedSettings())
    with pytest.raises(ResultsError, match="no result rows"):
        summarize_export([], settings=IsolatedSettings())
    with pytest.raises(ResultsError, match="no results list"):
        summarize_export({"results": {}}, settings=IsolatedSettings())
    with pytest.raises(ResultsError, match="no case id or system"):
        summarize_export([{"latencyMs": 1}], settings=IsolatedSettings())


def test_write_summary_round_trip(tmp_path: Path) -> None:
    summary = summarize_export(_wrapped(_pair()), settings=IsolatedSettings())
    markdown_path, json_path = write_summary(summary, tmp_path / "full.md")
    assert markdown_path.name == "full.md"
    assert json_path.name == "full.json"
    loaded = EvalSummary.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert loaded.case_ids == summary.case_ids
    assert loaded.baseline.judge_mean == pytest.approx(summary.baseline.judge_mean)
    assert markdown_path.read_text(encoding="utf-8").endswith("\n")


def test_smoke_cases_cover_the_checks_a_string_can_fail() -> None:
    brands: set[str] = set()
    channels: set[str] = set()
    mentions = False
    capped = False
    uncapped = False
    assert len(SMOKE_CASE_IDS) == len(set(SMOKE_CASE_IDS)) == 5
    for case_id in SMOKE_CASE_IDS:
        case = load_case(case_id)
        brands.add(case.brand_id)
        channels.update(case.brief.channels)
        if case.hard_constraints.must_mention:
            mentions = True
        if case.hard_constraints.headline_max_chars:
            capped = True
        else:
            uncapped = True
    assert brands == {"brightleaf", "ledgerly", "voltride"}
    assert channels == {"search", "social", "display", "email"}
    assert mentions
    assert capped
    assert uncapped


def test_smoke_config_matches_the_full_run_except_the_case_list() -> None:
    root = CASES_DIR.parent
    full = yaml.safe_load((root / "promptfooconfig.yaml").read_text(encoding="utf-8"))
    smoke = yaml.safe_load((root / "promptfooconfig.smoke.yaml").read_text(encoding="utf-8"))
    assert isinstance(full, dict)
    assert isinstance(smoke, dict)
    for key in ("prompts", "providers", "defaultTest", "commandLineOptions"):
        assert smoke[key] == full[key]
    tests = smoke["tests"]
    assert isinstance(tests, list)
    assert tests[0]["config"]["case_ids"] == list(SMOKE_CASE_IDS)


def test_cli_writes_markdown_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())
    export = tmp_path / "export.json"
    export.write_text(json.dumps(_wrapped(_pair())), encoding="utf-8")
    stem = tmp_path / "out" / "full"
    result = runner.invoke(cli.app, ["eval-summary", str(export), "--output", str(stem)])
    assert result.exit_code == 0
    assert "Judge mean" in result.output
    loaded = EvalSummary.model_validate_json(stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert loaded.baseline.rows == 2
    assert "Wrote" in result.stderr


def test_cli_rejects_an_empty_export_and_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())
    export = tmp_path / "empty.json"
    export.write_text("[]", encoding="utf-8")
    result = runner.invoke(cli.app, ["eval-summary", str(export)])
    assert result.exit_code == 1
    assert "no result rows" in result.output

    usable = tmp_path / "one.json"
    usable.write_text(json.dumps(_wrapped(_pair()[:1])), encoding="utf-8")
    refused = runner.invoke(cli.app, ["eval-summary", str(usable), "--output", str(tmp_path)])
    assert refused.exit_code == 1
    assert "directory" in refused.output


def test_help_lists_eval_summary() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "eval-summary" in result.output


_FULL_RESULTS = CASES_DIR.parent / "results" / "full.json"


@pytest.mark.skipif(
    not _FULL_RESULTS.is_file(),
    reason="manual 30-case summary is not in evals/results/ yet",
)
def test_committed_full_run_covers_the_dataset() -> None:
    """The manual 30-case run, once evals/results/full.json has been written.

    `uv run poe check` does not create that file. The command is in docs/promptfoo.md.
    """
    path = _FULL_RESULTS
    summary = EvalSummary.model_validate_json(path.read_text(encoding="utf-8"))
    case_ids = list_case_ids()
    assert summary.kind == "full"
    assert summary.case_ids == case_ids
    assert summary.baseline.rows == len(case_ids)
    assert summary.pipeline.rows == len(case_ids)
    systems = {(row.system, row.case_id) for row in summary.rows}
    assert systems == {("baseline", case_id) for case_id in case_ids} | {
        ("pipeline", case_id) for case_id in case_ids
    }
    assert summary.baseline.errors == sum(
        1 for row in summary.rows if row.system == "baseline" and row.error is not None
    )
    assert summary.pipeline.deterministic_passes == sum(
        1 for row in summary.rows if row.system == "pipeline" and row.deterministic_pass
    )
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert markdown.startswith("# Full eval run\n")
    for case_id in case_ids:
        assert case_id in markdown
