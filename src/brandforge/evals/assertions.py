"""Deterministic checks on an eval row (BF-35).

promptfoo runs these after each provider returns. pytest runs the same functions,
so a check does not depend on the Node runner and does not call a model. The rules
are in ADR 0028.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from brandforge.evals.cases import EvalCase, load_case
from brandforge.evals.promptfoo import EvalOutput, case_id_from_context
from brandforge.models import BrandProfile

_CHANNELS: frozenset[str] = frozenset({"search", "social", "display", "email"})
_COPY_FIELDS = ("headline", "body", "cta")
_FIELD_NAMES = {"headline": "headline", "body": "body", "cta": "call to action"}


@dataclass(frozen=True)
class Check:
    """One deterministic check. ``passed`` is all or nothing: the score is 1 or 0."""

    passed: bool
    reason: str


def check_json(output: str, *, case: EvalCase) -> Check:
    """The output parses as the eval row for this case and brand.

    Syntax is only half of it. A JSON value that is not an ``EvalOutput``, or that
    names another case, cannot be compared with the rest of the run.
    """
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return _fail(f"output is not valid JSON ({exc.msg})")
    try:
        row = EvalOutput.model_validate(payload)
    except ValidationError as exc:
        return _fail(f"output is not the eval row: {_schema_reason(exc)}")
    problems: list[str] = []
    if row.case_id != case.id:
        problems.append(f"row case_id is {row.case_id!r}; this test is {case.id!r}")
    if row.brand_id != case.brand_id:
        problems.append(f"row brand_id is {row.brand_id!r}; this test is {case.brand_id!r}")
    if problems:
        return _fail("; ".join(problems))
    return _ok(f"eval row for {case.id}")


def check_channel_limits(output: str, *, case: EvalCase) -> Check:
    """Every requested channel is present, and stated headline caps hold.

    A channel with no entry in ``headline_max_chars`` is not length-checked. The
    email subject is the headline. The length is ``len`` of the headline string,
    in Unicode code points. Every variant of a capped channel has to fit. A variant
    for a channel the brief did not request fails, and so does a missing channel.
    """
    variants = _read(output)
    if isinstance(variants, str):
        return _fail(variants)
    problems: list[str] = []
    seen: set[str] = set()
    requested = set(case.brief.channels)
    for index, item in enumerate(variants):
        label = _label(item, index)
        channel, problem = _channel(item, label)
        if problem is not None:
            problems.append(problem)
            continue
        assert channel is not None
        if channel not in requested:
            problems.append(f"{label} channel is not in the brief")
            continue
        seen.add(channel)
        headline, problem = _string_field(item, "headline", label)
        if problem is not None:
            problems.append(problem)
            continue
        assert headline is not None
        cap = _cap(case, channel)
        if cap is not None and len(headline) > cap:
            problems.append(f"{label} headline is {len(headline)} characters; the limit is {cap}")
    missing = sorted(requested - seen)
    if len(missing) == 1:
        problems.append(f"missing channel: {missing[0]}")
    elif missing:
        problems.append(f"missing channels: {', '.join(missing)}")
    if problems:
        return _fail("; ".join(problems))
    return _ok("every requested channel is present and headlines are within the stated caps")


def check_banned_words(output: str, *, brand: BrandProfile) -> Check:
    """No brand banned word appears in a headline, body, or call to action.

    The match is a case-insensitive substring, the same rule as the example corpus.
    Words come from the brand profile, not from the case file.
    """
    variants = _read(output)
    if isinstance(variants, str):
        return _fail(variants)
    problems: list[str] = []
    for index, item in enumerate(variants):
        label = _label(item, index)
        text, problem = _copy(item, label)
        if problem is not None or text is None:
            problems.append(problem or f"{label} has no copy")
            continue
        folded = text.casefold()
        for banned in brand.banned_words:
            if banned.casefold() in folded:
                problems.append(f"{label} uses banned word {banned!r}")
    if problems:
        return _fail("; ".join(problems))
    return _ok("no banned words")


def check_cta(output: str) -> Check:
    """Every variant has a call to action, and the row has at least one variant."""
    variants = _read(output)
    if isinstance(variants, str):
        return _fail(variants)
    problems: list[str] = []
    for index, item in enumerate(variants):
        label = _label(item, index)
        cta, problem = _string_field(item, "cta", label)
        if problem is not None:
            problems.append(problem)
            continue
        assert cta is not None
        if not cta.strip():
            problems.append(f"{label} call to action is blank")
    if problems:
        return _fail("; ".join(problems))
    return _ok("every variant has a call to action")


def check_must_mention(output: str, *, case: EvalCase) -> Check:
    """Every variant includes each required phrase in its headline, body, or call to action.

    A case with no ``must_mention`` phrases passes once it has variants. The match is
    a case-insensitive substring. A phrase split across two fields does not count.
    """
    variants = _read(output)
    if isinstance(variants, str):
        return _fail(variants)
    phrases = list(case.hard_constraints.must_mention)
    problems: list[str] = []
    for index, item in enumerate(variants):
        label = _label(item, index)
        text, problem = _copy(item, label)
        if problem is not None or text is None:
            problems.append(problem or f"{label} has no copy")
            continue
        folded = text.casefold()
        for phrase in phrases:
            if phrase.casefold() not in folded:
                problems.append(f"{label} does not mention {phrase!r}")
    if problems:
        return _fail("; ".join(problems))
    if phrases:
        return _ok("every variant includes the required phrases")
    return _ok("no required phrases")


def assert_json(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """promptfoo entry for :func:`check_json`. A bad case is a failed check, not a raise."""
    return _entry(output, context, _json_for_case)


def assert_channel_limits(
    output: object, context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """promptfoo entry for :func:`check_channel_limits`."""
    return _entry(output, context, _limits_for_case)


def assert_banned_words(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """promptfoo entry for :func:`check_banned_words`. The brand is the case's profile."""
    return _entry(output, context, _banned_for_brand, load_brand_profile=True)


def assert_cta(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """promptfoo entry for :func:`check_cta`. The case is not required."""
    del context
    try:
        return _grade(check_cta(_coerce(output)))
    except AssertionError:
        raise
    except Exception as exc:
        return _grade(_fail(_error(exc)))


def assert_must_mention(output: object, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """promptfoo entry for :func:`check_must_mention`."""
    return _entry(output, context, _mentions_for_case)


_Checker = Callable[[str, EvalCase, BrandProfile | None], Check]


def _entry(
    output: object,
    context: Mapping[str, Any] | None,
    check: _Checker,
    *,
    load_brand_profile: bool = False,
) -> dict[str, Any]:
    try:
        case = _case(context)
        brand = _load_brand(case) if load_brand_profile else None
        return _grade(check(_coerce(output), case, brand))
    except AssertionError:
        raise
    except Exception as exc:
        return _grade(_fail(_error(exc)))


def _case(context: Mapping[str, Any] | None) -> EvalCase:
    prompt = None if context is None else context.get("prompt")
    return load_case(case_id_from_context(prompt, context))


def _json_for_case(text: str, case: EvalCase, brand: BrandProfile | None) -> Check:
    del brand
    return check_json(text, case=case)


def _limits_for_case(text: str, case: EvalCase, brand: BrandProfile | None) -> Check:
    del brand
    return check_channel_limits(text, case=case)


def _banned_for_brand(text: str, case: EvalCase, brand: BrandProfile | None) -> Check:
    del case
    if brand is None:
        return _fail("brand profile was not loaded")
    return check_banned_words(text, brand=brand)


def _mentions_for_case(text: str, case: EvalCase, brand: BrandProfile | None) -> Check:
    del brand
    return check_must_mention(text, case=case)


def _load_brand(case: EvalCase) -> BrandProfile:
    from brandforge.brands import load_brand

    return load_brand(case.brand_id)


def _read(output: str) -> list[dict[str, Any]] | str:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return f"output is not valid JSON ({exc.msg})"
    if not isinstance(payload, dict):
        return "output is not a JSON object"
    raw = payload.get("variants")
    if not isinstance(raw, list):
        return "output has no variants list"
    if not raw:
        return "output has no variants"
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            return f"variant {index} is not an object"
        items.append(item)
    return items


def _label(item: Mapping[str, Any], index: int) -> str:
    channel = item.get("channel")
    ident = item.get("id")
    channel_bit = channel.strip() if isinstance(channel, str) and channel.strip() else None
    id_bit = ident.strip() if isinstance(ident, str) and ident.strip() else None
    if channel_bit and id_bit:
        return f"{channel_bit} {id_bit}"
    if id_bit:
        return id_bit
    return f"variant {index}"


def _channel(item: Mapping[str, Any], label: str) -> tuple[str | None, str | None]:
    if "channel" not in item or item["channel"] is None:
        return None, f"{label} has no channel"
    channel = item["channel"]
    if not isinstance(channel, str):
        return None, f"{label} channel is not text"
    if channel not in _CHANNELS:
        return None, f"{label} channel {channel!r} is not a channel"
    return channel, None


def _string_field(item: Mapping[str, Any], key: str, label: str) -> tuple[str | None, str | None]:
    name = _FIELD_NAMES[key]
    if key not in item or item[key] is None:
        return None, f"{label} has no {name}"
    value = item[key]
    if not isinstance(value, str):
        return None, f"{label} {name} is not text"
    return value, None


def _copy(item: Mapping[str, Any], label: str) -> tuple[str | None, str | None]:
    parts: list[str] = []
    for key in _COPY_FIELDS:
        value, problem = _string_field(item, key, label)
        if problem is not None:
            return None, problem
        assert value is not None
        parts.append(value)
    return "\n".join(parts), None


def _cap(case: EvalCase, channel: str) -> int | None:
    for name, limit in case.hard_constraints.headline_max_chars.items():
        if name == channel:
            return limit
    return None


def _schema_reason(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        message = str(err["msg"])
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) if parts else "schema validation failed"


def _coerce(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _grade(check: Check) -> dict[str, Any]:
    return {"pass": check.passed, "score": 1.0 if check.passed else 0.0, "reason": check.reason}


def _ok(reason: str) -> Check:
    return Check(passed=True, reason=reason)


def _fail(reason: str) -> Check:
    return Check(passed=False, reason=reason)


def _error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
