"""RunState, its reducers and the contracts it introduces (Plan, Example, RunError)."""

import pytest
from pydantic import ValidationError

from brandforge.brands import load_brand
from brandforge.models import (
    Brief,
    Example,
    Plan,
    RunError,
    Usage,
    add_errors,
    add_usage,
    new_run_state,
)


@pytest.fixture
def brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search", "social"],
    )


def test_add_usage_sums_tokens_and_cost() -> None:
    total = add_usage(
        Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
        Usage(input_tokens=20, output_tokens=15, cost_usd=0.002),
    )

    assert total.input_tokens == 30
    assert total.output_tokens == 20
    assert total.cost_usd == pytest.approx(0.003)


def test_add_errors_appends_without_mutating_inputs() -> None:
    first = [RunError(node="writer", message="timeout")]
    second = [RunError(node="critic", message="bad schema")]

    merged = add_errors(first, second)

    assert [e.node for e in merged] == ["writer", "critic"]
    assert len(first) == 1
    assert len(second) == 1


def test_new_run_state_starts_empty_and_running(brief: Brief) -> None:
    brand = load_brand("voltride")

    state = new_run_state(brief, brand)

    assert state["brief"] is brief
    assert state["brand"] is brand
    assert state["status"] == "running"
    assert state["plan"] is None
    assert state["examples"] == []
    assert state["variants"] == []
    assert state["critiques"] == []
    assert state["revision_count"] == 0
    assert state["errors"] == []
    assert state["usage"] == Usage()


def test_new_run_state_ids_are_unique_unless_given(brief: Brief) -> None:
    brand = load_brand("voltride")

    assert new_run_state(brief, brand)["run_id"] != new_run_state(brief, brand)["run_id"]
    assert new_run_state(brief, brand, run_id="run-1")["run_id"] == "run-1"


def test_plan_requires_channels_and_a_sane_variant_count() -> None:
    ok = Plan(audience="commuters", angle="save time", channels=["search"], variants_per_channel=3)
    assert ok.variants_per_channel == 3

    with pytest.raises(ValidationError):
        Plan(audience="commuters", angle="save time", channels=[], variants_per_channel=3)
    with pytest.raises(ValidationError):
        Plan(audience="commuters", angle="save time", channels=["search"], variants_per_channel=0)


def test_example_and_run_error_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Example.model_validate(
            {
                "brand_id": "voltride",
                "channel": "search",
                "headline": "h",
                "body": "b",
                "cta": "c",
                "extra": 1,
            }
        )
    with pytest.raises(ValidationError):
        RunError(node="  ", message="boom")
