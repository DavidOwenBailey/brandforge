"""Eval dataset (BF-33), promptfoo providers (BF-34), and deterministic checks (BF-35).

The YAML cases live in the repo's ``evals/cases/`` directory. promptfoo loads
one test per case, runs that case through the baseline and the pipeline, and
scores the row without another model call.
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
from brandforge.evals.cases import (
    CaseLoadError,
    EvalCase,
    HardConstraints,
    list_case_ids,
    load_case,
    load_cases,
)
from brandforge.evals.promptfoo import (
    EvalOutput,
    EvalUsage,
    EvalVariant,
    call_baseline,
    call_pipeline,
    generate_tests,
)

__all__ = [
    "CaseLoadError",
    "Check",
    "EvalCase",
    "EvalOutput",
    "EvalUsage",
    "EvalVariant",
    "HardConstraints",
    "assert_banned_words",
    "assert_channel_limits",
    "assert_cta",
    "assert_json",
    "assert_must_mention",
    "call_baseline",
    "call_pipeline",
    "check_banned_words",
    "check_channel_limits",
    "check_cta",
    "check_json",
    "check_must_mention",
    "generate_tests",
    "list_case_ids",
    "load_case",
    "load_cases",
]
