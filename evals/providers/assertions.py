"""promptfoo assertions for an eval row (BF-35)."""

from collections.abc import Mapping
from typing import Any

from brandforge.evals import assertions


def assert_json(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The output is the eval row for this case. promptfoo calls this."""
    return assertions.assert_json(output, context)


def assert_channel_limits(
    output: object, context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Headline caps and the requested channels. promptfoo calls this."""
    return assertions.assert_channel_limits(output, context)


def assert_banned_words(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Brand banned words are absent. promptfoo calls this."""
    return assertions.assert_banned_words(output, context)


def assert_cta(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every variant has a call to action. promptfoo calls this."""
    return assertions.assert_cta(output, context)


def assert_must_mention(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every variant includes the required phrases. promptfoo calls this."""
    return assertions.assert_must_mention(output, context)
