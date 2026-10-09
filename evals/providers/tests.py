"""promptfoo test generator: one row per eval case (BF-34)."""

from collections.abc import Mapping
from typing import Any

from brandforge.evals.promptfoo import generate_tests as _generate_tests


def generate_tests(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the promptfoo tests. promptfoo calls this."""
    return _generate_tests(config)
