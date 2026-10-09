"""Eval dataset loader (BF-33).

The YAML cases live in the repo's ``evals/cases/`` directory, next to the
promptfoo config the later eval tasks add. This package only loads them.
"""

from brandforge.evals.cases import (
    CaseLoadError,
    EvalCase,
    HardConstraints,
    list_case_ids,
    load_case,
    load_cases,
)

__all__ = [
    "CaseLoadError",
    "EvalCase",
    "HardConstraints",
    "list_case_ids",
    "load_case",
    "load_cases",
]
