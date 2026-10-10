"""promptfoo llm-rubric grader for an eval row (BF-36)."""

from collections.abc import Mapping
from typing import Any

from brandforge.evals.judge import call_judge


def call_api(
    prompt: str,
    options: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one eval row on the judge rubric. promptfoo calls this."""
    return call_judge(prompt, options, context)
