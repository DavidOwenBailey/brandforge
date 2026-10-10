"""Deterministic eval checks. No model calls."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from brandforge.brands import load_brand
from brandforge.evals import (
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
    list_case_ids,
    load_case,
)
from brandforge.evals.promptfoo import EvalOutput

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
CASE_ID = "brightleaf_01_spring_blossom"
MENTION_ID = "voltride_04_test_ride_weekends"
_PASS = "every requested channel is present and headlines are within the stated caps"


def _variant(case_id: str, channel: str, **overrides: Any) -> dict[str, Any]:
    case = load_case(case_id)
    phrases = " ".join(case.hard_constraints.must_mention)
    item: dict[str, Any] = {
        "id": f"{channel}-1",
        "channel": channel,
        "headline": "Hello",
        "body": phrases or "A plain line.",
        "cta": "Read more",
        "flagged": None,
    }
    item.update(overrides)
    return item


def _row(case_id: str, variants: list[dict[str, Any]] | None = None, **overrides: Any) -> str:
    case = load_case(case_id)
    brand = load_brand(case.brand_id)
    if variants is None:
        variants = [_variant(case_id, channel) for channel in case.brief.channels]
    payload: dict[str, Any] = {
        "system": "baseline",
        "case_id": case.id,
        "brand_id": case.brand_id,
        "brand_version": brand.version,
        "rubric_version": brand.rubric.version,
        "status": "complete",
        "run_id": None,
        "trace_id": None,
        "revision_count": 0,
        "variants": variants,
        "errors": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 2,
            "cost_usd": 0.0,
        },
    }
    payload.update(overrides)
    return json.dumps(payload)


def _failed(check: Check) -> str:
    assert check.passed is False
    return check.reason


@pytest.mark.parametrize("case_id", list_case_ids())
def test_compliant_copy_passes_every_check(case_id: str) -> None:
    case = load_case(case_id)
    brand = load_brand(case.brand_id)
    output = _row(case_id)
    EvalOutput.model_validate_json(output)
    assert check_json(output, case=case).reason == f"eval row for {case_id}"
    assert check_channel_limits(output, case=case).reason == _PASS
    assert check_banned_words(output, brand=brand).reason == "no banned words"
    assert check_cta(output).reason == "every variant has a call to action"
    mention = check_must_mention(output, case=case)
    if case.hard_constraints.must_mention:
        assert mention.reason == "every variant includes the required phrases"
    else:
        assert mention.reason == "no required phrases"


def test_a_headline_on_the_cap_passes_and_one_past_it_fails() -> None:
    case = load_case(CASE_ID)
    on_cap = _row(
        CASE_ID,
        variants=[
            _variant(CASE_ID, "email", headline="é" * 50),
            _variant(CASE_ID, "social", headline="y" * 200),
        ],
    )
    assert check_channel_limits(on_cap, case=case).passed is True

    over = _row(
        CASE_ID,
        variants=[
            _variant(CASE_ID, "email", id="email-1", headline="x" * 50),
            _variant(CASE_ID, "email", id="email-2", headline="x" * 51),
            _variant(CASE_ID, "social"),
        ],
    )
    assert _failed(check_channel_limits(over, case=case)) == (
        "email email-2 headline is 51 characters; the limit is 50"
    )


def test_a_missing_channel_and_an_extra_channel_both_fail() -> None:
    case = load_case(CASE_ID)
    output = _row(
        CASE_ID,
        variants=[_variant(CASE_ID, "social"), _variant(CASE_ID, "display", id="display-1")],
    )
    assert _failed(check_channel_limits(output, case=case)) == (
        "display display-1 channel is not in the brief; missing channel: email"
    )


def test_every_requested_channel_is_named_when_several_are_missing() -> None:
    case = load_case(CASE_ID)
    output = _row(CASE_ID, variants=[_variant(CASE_ID, "display")])
    assert "missing channels: email, social" in _failed(check_channel_limits(output, case=case))


def test_an_unknown_channel_is_not_measured_against_a_cap() -> None:
    case = load_case(CASE_ID)
    output = _row(CASE_ID, variants=[_variant(CASE_ID, "billboard")])
    reason = _failed(check_channel_limits(output, case=case))
    assert "channel 'billboard' is not a channel" in reason
    assert "limit is" not in reason


def test_banned_words_match_the_brand_profile_case_insensitively() -> None:
    case = load_case(CASE_ID)
    brand = load_brand("brightleaf")
    output = _row(
        CASE_ID,
        variants=[
            _variant(CASE_ID, "email", body="A miracle detox tea"),
            _variant(CASE_ID, "social", body="A plain line."),
        ],
    )
    assert _failed(check_banned_words(output, brand=brand)) == (
        "email email-1 uses banned word 'detox'; email email-1 uses banned word 'miracle'"
    )
    longer = _row(
        CASE_ID,
        variants=[
            _variant(CASE_ID, "email", body="A detoxify blend"),
            _variant(CASE_ID, "social"),
        ],
    )
    assert "banned word 'detox'" in _failed(check_banned_words(longer, brand=brand))

    other = load_brand("voltride")
    guaranteed = _row(
        CASE_ID,
        variants=[
            _variant(CASE_ID, "email", body="A guaranteed cup"),
            _variant(CASE_ID, "social"),
        ],
    )
    assert check_banned_words(guaranteed, brand=brand).passed is True
    assert "banned word 'guaranteed'" in _failed(check_banned_words(guaranteed, brand=other))
    assert check_json(output, case=case).passed is True


def test_a_blank_call_to_action_fails_and_a_padded_one_passes() -> None:
    case = load_case(CASE_ID)
    blank = _row(
        CASE_ID,
        variants=[
            _variant(CASE_ID, "email", cta="  "),
            _variant(CASE_ID, "social"),
        ],
    )
    assert _failed(check_cta(blank)) == "email email-1 call to action is blank"
    assert check_json(blank, case=case).passed is False

    missing = json.loads(_row(CASE_ID))
    del missing["variants"][0]["cta"]
    assert "has no call to action" in _failed(check_cta(json.dumps(missing)))

    padded = _row(
        CASE_ID,
        variants=[
            _variant(CASE_ID, "email", cta="  Read more  "),
            _variant(CASE_ID, "social"),
        ],
    )
    assert check_cta(padded).passed is True
    assert check_json(padded, case=case).passed is True


def test_a_required_phrase_must_appear_in_every_variant() -> None:
    case = load_case(MENTION_ID)
    in_cta = _row(
        MENTION_ID,
        variants=[
            _variant(MENTION_ID, channel, body="A plain line.", cta="Book a Test Ride")
            for channel in case.brief.channels
        ],
    )
    assert check_must_mention(in_cta, case=case).passed is True

    split = _row(
        MENTION_ID,
        variants=[
            _variant(MENTION_ID, "email", headline="test", body="ride", cta="Book"),
            _variant(MENTION_ID, "social"),
            _variant(MENTION_ID, "search"),
        ],
    )
    assert _failed(check_must_mention(split, case=case)) == (
        "email email-1 does not mention 'test ride'"
    )

    one_missing = _row(
        MENTION_ID,
        variants=[
            _variant(MENTION_ID, "email", id="email-1"),
            _variant(MENTION_ID, "social", id="social-9", body="Come along"),
            _variant(MENTION_ID, "search"),
        ],
    )
    assert _failed(check_must_mention(one_missing, case=case)) == (
        "social social-9 does not mention 'test ride'"
    )


def test_a_case_with_no_required_phrase_passes_that_check() -> None:
    case = load_case(CASE_ID)
    assert case.hard_constraints.must_mention == []
    assert check_must_mention(_row(CASE_ID), case=case).reason == "no required phrases"


def test_json_checks_the_schema_and_the_case_identity() -> None:
    case = load_case(CASE_ID)
    output = json.loads(_row(CASE_ID))
    output["status"] = "partial"
    partial = json.dumps(output)
    assert check_json(partial, case=case).passed is True
    assert check_channel_limits(partial, case=case).passed is True

    output["status"] = "nope"
    broken = json.dumps(output)
    assert "status" in _failed(check_json(broken, case=case))
    assert check_cta(broken).passed is True
    assert check_channel_limits(broken, case=case).passed is True

    output["status"] = "complete"
    output["case_id"] = MENTION_ID
    output["brand_id"] = "voltride"
    renamed = json.dumps(output)
    assert _failed(check_json(renamed, case=case)) == (
        f"row case_id is {MENTION_ID!r}; this test is {CASE_ID!r}; "
        "row brand_id is 'voltride'; this test is 'brightleaf'"
    )
    assert check_banned_words(renamed, brand=load_brand("brightleaf")).passed is True


def test_an_empty_or_unreadable_row_fails_the_content_checks() -> None:
    case = load_case(CASE_ID)
    brand = load_brand(case.brand_id)
    empty = _row(CASE_ID, variants=[])
    assert check_json(empty, case=case).passed is True
    for check in (
        check_channel_limits(empty, case=case),
        check_banned_words(empty, brand=brand),
        check_cta(empty),
        check_must_mention(empty, case=case),
    ):
        assert check.reason == "output has no variants"

    for check in (
        check_json("not json", case=case),
        check_channel_limits("not json", case=case),
        check_banned_words("not json", brand=brand),
        check_cta("not json"),
        check_must_mention("not json", case=case),
    ):
        assert check.reason == "output is not valid JSON (Expecting value)"

    assert check_channel_limits("[]", case=case).reason == "output is not a JSON object"
    assert check_cta('{"variants": "nope"}').reason == "output has no variants list"
    assert check_cta('{"variants": [1]}').reason == "variant 0 is not an object"


def test_promptfoo_entries_load_the_case_and_do_not_call_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("model called")

    monkeypatch.setattr("brandforge.llm.gateway.complete_structured", boom)
    monkeypatch.setattr("brandforge.baseline.generate_baseline", boom)
    monkeypatch.setattr("brandforge.graph.run_graph", boom)

    output = _row(CASE_ID)
    context = {"vars": {"case_id": f"  {CASE_ID}  "}, "prompt": MENTION_ID}
    assert assert_json(output, context) == {
        "pass": True,
        "score": 1.0,
        "reason": f"eval row for {CASE_ID}",
    }
    assert assert_channel_limits(output, {"prompt": CASE_ID})["pass"] is True
    assert assert_banned_words(output, {"vars": {"case_id": CASE_ID}})["pass"] is True
    assert assert_must_mention(output, {"vars": {"case_id": CASE_ID}})["pass"] is True
    assert assert_cta(json.loads(output), None)["pass"] is True


def test_a_missing_case_fails_the_assertion_instead_of_raising() -> None:
    result = assert_json("{}", {"vars": {"case_id": "no-such-case"}})
    assert result["pass"] is False
    assert result["score"] == 0.0
    assert result["reason"].startswith("CaseLoadError:")
    missing = assert_channel_limits(_row(CASE_ID), {})
    assert missing["pass"] is False
    assert "case_id" in missing["reason"]


def test_an_assertion_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(prompt: object, context: object) -> str:
        del prompt, context
        raise AssertionError("bug")

    monkeypatch.setattr("brandforge.evals.assertions.case_id_from_context", boom)
    with pytest.raises(AssertionError, match="bug"):
        assert_json("{}", {"vars": {"case_id": CASE_ID}})


def test_the_promptfoo_script_forwards_to_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def fake(output: object, context: object = None) -> dict[str, Any]:
        seen.append((output, context))
        return {"pass": True, "score": 1.0, "reason": "forwarded"}

    monkeypatch.setattr("brandforge.evals.assertions.assert_json", fake)
    monkeypatch.setattr("brandforge.evals.assertions.assert_channel_limits", fake)
    monkeypatch.setattr("brandforge.evals.assertions.assert_banned_words", fake)
    monkeypatch.setattr("brandforge.evals.assertions.assert_cta", fake)
    monkeypatch.setattr("brandforge.evals.assertions.assert_must_mention", fake)
    script = _load_script(EVALS / "providers" / "assertions.py")
    context = {"vars": {"case_id": CASE_ID}}
    assert script.assert_json("row", context)["reason"] == "forwarded"
    assert script.assert_channel_limits("row", context)["reason"] == "forwarded"
    assert script.assert_banned_words("row", context)["reason"] == "forwarded"
    assert script.assert_cta("row", context)["reason"] == "forwarded"
    assert script.assert_must_mention("row", context)["reason"] == "forwarded"
    assert seen == [("row", context)] * 5


def _load_script(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"promptfoo_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
