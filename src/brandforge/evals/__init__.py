"""Eval dataset (BF-33), promptfoo providers (BF-34), checks (BF-35), judge (BF-36),
calibration (BF-37) and the run summary (BF-38).

The YAML cases live in the repo's ``evals/cases/`` directory. promptfoo loads
one test per case, runs that case through the baseline and the pipeline, scores
the row with deterministic checks, then grades each variant on the judge rubric.
``evals/calibration/`` holds hand scores for 15 outputs. ``calibrate`` compares
them with that judge. ``summarize_path`` turns a promptfoo export into the
side-by-side summary committed under ``evals/results/``.
"""

from brandforge.evals.assertions import (
    Check,
    assert_banned_words,
    assert_channel_limits,
    assert_cta,
    assert_json,
    assert_must_mention,
    check_banned_words,
    check_channel_limits,
    check_cta,
    check_json,
    check_must_mention,
)
from brandforge.evals.calibration import (
    CalibrationError,
    CalibrationItem,
    CalibrationReport,
    calibrate,
    format_report,
    load_calibration,
    run_calibration,
)
from brandforge.evals.cases import (
    CaseLoadError,
    EvalCase,
    HardConstraints,
    list_case_ids,
    load_case,
    load_cases,
)
from brandforge.evals.judge import (
    JudgeGrade,
    JudgeReply,
    anchored_rubric,
    call_judge,
    judge_assertions,
    score_row,
)
from brandforge.evals.promptfoo import (
    EvalOutput,
    EvalUsage,
    EvalVariant,
    call_baseline,
    call_pipeline,
    generate_tests,
)
from brandforge.evals.results import (
    SMOKE_CASE_IDS,
    EvalSummary,
    ResultsError,
    format_summary,
    summarize_path,
)

__all__ = [
    "SMOKE_CASE_IDS",
    "CalibrationError",
    "CalibrationItem",
    "CalibrationReport",
    "CaseLoadError",
    "Check",
    "EvalCase",
    "EvalOutput",
    "EvalSummary",
    "EvalUsage",
    "EvalVariant",
    "HardConstraints",
    "JudgeGrade",
    "JudgeReply",
    "ResultsError",
    "anchored_rubric",
    "assert_banned_words",
    "assert_channel_limits",
    "assert_cta",
    "assert_json",
    "assert_must_mention",
    "calibrate",
    "call_baseline",
    "call_judge",
    "call_pipeline",
    "check_banned_words",
    "check_channel_limits",
    "check_cta",
    "check_json",
    "check_must_mention",
    "format_report",
    "format_summary",
    "generate_tests",
    "judge_assertions",
    "list_case_ids",
    "load_calibration",
    "load_case",
    "load_cases",
    "run_calibration",
    "score_row",
    "summarize_path",
]
