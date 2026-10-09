"""Eval dataset (BF-33) and the promptfoo providers that run it (BF-34).

The YAML cases live in the repo's ``evals/cases/`` directory. promptfoo loads
one test per case and runs that case through the baseline and the pipeline.
"""

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
    "EvalCase",
    "EvalOutput",
    "EvalUsage",
    "EvalVariant",
    "HardConstraints",
    "call_baseline",
    "call_pipeline",
    "generate_tests",
    "list_case_ids",
    "load_case",
    "load_cases",
]
