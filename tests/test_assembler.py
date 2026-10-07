"""Assembler tests. It is plain code, so every case is hand-built state."""

import pytest

from brandforge.agents.assembler import assemble_result
from brandforge.brands.loader import load_brand
from brandforge.models import (
    Brief,
    Critique,
    RunError,
    RunResult,
    RunState,
    Usage,
    Variant,
    new_run_state,
)


def _variant(variant_id: str, channel: str = "search") -> Variant:
    return Variant.model_validate(
        {"id": variant_id, "channel": channel, "headline": "H", "body": "B", "cta": "C"}
    )


def _critique(variant_id: str, *, passed: bool, fixes: list[str] | None = None) -> Critique:
    return Critique(
        variant_id=variant_id,
        scores={"voice": 5 if passed else 2, "clarity": 4},
        overall=4.5 if passed else 3.0,
        passed=passed,
        fixes=fixes or [],
    )


def _state(
    results: dict[str, bool | None],
    *,
    errors: list[RunError] | None = None,
    revision_count: int = 0,
) -> RunState:
    """One variant per entry in `results` (id -> passed). `None` means it was never scored."""
    brief = Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )
    state = new_run_state(brief, load_brand("voltride"), run_id="run-1")
    state["variants"] = [_variant(variant_id) for variant_id in results]
    state["critiques"] = [
        _critique(variant_id, passed=passed)
        for variant_id, passed in results.items()
        if passed is not None
    ]
    state["errors"] = errors or []
    state["revision_count"] = revision_count
    return state


def _result(update: dict[str, object]) -> RunResult:
    result = update["result"]
    assert isinstance(result, RunResult)
    return result


def test_all_variants_passing_is_complete_with_nothing_flagged() -> None:
    update = assemble_result(_state({"a": True, "b": True}))

    result = _result(update)
    assert result.status == "complete"
    assert [item.variant.id for item in result.variants] == ["a", "b"]
    assert [item.flagged for item in result.variants] == [False, False]
    assert result.flagged_count == 0


def test_a_failing_variant_is_flagged_and_the_run_is_partial() -> None:
    update = assemble_result(_state({"a": True, "b": False}))

    result = _result(update)
    assert result.status == "partial"
    assert [item.flagged for item in result.variants] == [False, True]
    assert result.flagged_count == 1


def test_a_variant_that_was_never_scored_is_flagged_without_a_critique() -> None:
    update = assemble_result(_state({"a": True, "b": None}))

    result = _result(update)
    flagged = result.variants[1]
    assert flagged.flagged
    assert flagged.critique is None
    assert result.status == "partial"


def test_each_variant_carries_its_own_critique() -> None:
    update = assemble_result(_state({"a": True, "b": False}))

    result = _result(update)
    assert [item.critique.variant_id for item in result.variants if item.critique] == ["a", "b"]
    assert result.variants[0].critique is not None
    assert result.variants[0].critique.scores == {"voice": 5, "clarity": 4}
    assert result.variants[1].critique is not None
    assert result.variants[1].critique.passed is False


def test_the_critics_fixes_stay_with_a_flagged_variant() -> None:
    state = _state({"a": False})
    state["critiques"] = [_critique("a", passed=False, fixes=["Tighten the headline"])]

    result = _result(assemble_result(state))

    assert result.variants[0].critique is not None
    assert result.variants[0].critique.fixes == ["Tighten the headline"]


def test_recorded_errors_make_a_run_with_passing_variants_partial() -> None:
    error = RunError(node="writer", message="channel 'search': asked for 2 variants, got 1")

    result = _result(assemble_result(_state({"a": True}, errors=[error])))

    assert result.status == "partial"
    assert result.errors == [error]
    assert result.flagged_count == 0


def test_no_variants_is_failed_with_or_without_errors() -> None:
    error = RunError(node="writer", message="no variants")

    assert _result(assemble_result(_state({}))).status == "failed"
    assert _result(assemble_result(_state({}, errors=[error]))).status == "failed"


def test_the_update_sets_the_final_status_and_nothing_else() -> None:
    update = assemble_result(_state({"a": False}))

    assert set(update) == {"result", "status"}
    assert update["status"] == "partial"
    assert update["status"] == _result(update).status


def test_the_result_records_the_run_brand_versions_usage_and_revisions() -> None:
    state = _state({"a": True}, revision_count=2)
    state["usage"] = Usage(input_tokens=700, output_tokens=300, cost_usd=0.0123)
    brand = state["brand"]

    result = _result(assemble_result(state))

    assert result.run_id == "run-1"
    assert result.brand_id == brand.id
    assert result.brand_version == brand.version
    assert result.rubric_version == brand.rubric.version
    assert result.revision_count == 2
    assert result.usage.total_tokens == 1_000
    assert result.usage.cost_usd == pytest.approx(0.0123)


def test_assembling_does_not_change_the_state() -> None:
    state = _state({"a": False})
    before = {key: repr(value) for key, value in state.items()}

    assemble_result(state)

    assert {key: repr(value) for key, value in state.items()} == before


def test_the_result_survives_a_json_round_trip() -> None:
    result = _result(assemble_result(_state({"a": True, "b": None})))

    assert RunResult.model_validate_json(result.model_dump_json()) == result
