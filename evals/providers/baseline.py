"""promptfoo provider for the single-prompt baseline (BF-34)."""

from collections.abc import Mapping
from typing import Any

from brandforge.evals import promptfoo


def call_api(prompt: str, options: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Run one eval case through the baseline. promptfoo calls this."""
    return promptfoo.call_baseline(prompt, options, context)
