"""Router tests. The router is a pure function, so every case is hand-built state."""

from pydantic_settings import SettingsConfigDict

from brandforge.brands.loader import load_brand
from brandforge.config import Budgets, Settings
from brandforge.models import Brief, Critique, RunState, Usage, Variant, new_run_state
from brandforge.router import failing_variant_ids, route


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def _variant(variant_id: str) -> Variant:
    return Variant(id=variant_id, channel="search", headline="H", body="B", cta="C")


def _critique(variant_id: str, *, passed: bool) -> Critique:
    return Critique(
        variant_id=variant_id,
        scores={"voice": 5 if passed else 2},
        overall=5.0 if passed else 2.0,
        passed=passed,
        fixes=[] if passed else ["Tighten the headline"],
    )


def _state(
    results: dict[str, bool],
    *,
    revision_count: int = 0,
    tokens: int = 0,
) -> RunState:
    """State with one variant and critique per entry in `results` (id -> passed)."""
    brief = Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search"],
    )
    state = new_run_state(brief, load_brand("voltride"))
    state["variants"] = [_variant(variant_id) for variant_id in results]
    state["critiques"] = [_critique(vid, passed=passed) for vid, passed in results.items()]
    state["revision_count"] = revision_count
    state["usage"] = Usage(input_tokens=tokens)
    return state


def test_all_variants_passing_assembles() -> None:
    state = _state({"a": True, "b": True})

    assert route(state, settings=IsolatedSettings()) == "assemble"


def test_a_failing_variant_with_revisions_left_revises() -> None:
    state = _state({"a": True, "b": False})

    assert route(state, settings=IsolatedSettings()) == "revise"


def test_a_second_revision_is_allowed_below_the_cap() -> None:
    state = _state({"a": False}, revision_count=1)

    assert route(state, settings=IsolatedSettings()) == "revise"


def test_stops_once_the_revision_cap_is_reached() -> None:
    state = _state({"a": True, "b": False}, revision_count=2)

    assert route(state, settings=IsolatedSettings()) == "stop"


def test_the_cap_comes_from_settings() -> None:
    state = _state({"a": False}, revision_count=2)
    roomy = IsolatedSettings(budgets=Budgets(max_revisions=3))
    none_allowed = IsolatedSettings(budgets=Budgets(max_revisions=0))

    assert route(state, settings=roomy) == "revise"
    assert route(_state({"a": False}), settings=none_allowed) == "stop"


def test_stops_when_the_token_budget_is_reached() -> None:
    settings = IsolatedSettings(budgets=Budgets(max_tokens_per_run=1_000))

    assert route(_state({"a": False}, tokens=1_000), settings=settings) == "stop"
    assert route(_state({"a": False}, tokens=5_000), settings=settings) == "stop"


def test_revises_while_the_token_budget_has_room() -> None:
    settings = IsolatedSettings(budgets=Budgets(max_tokens_per_run=1_000))

    assert route(_state({"a": False}, tokens=999), settings=settings) == "revise"


def test_token_budget_counts_input_and_output_tokens() -> None:
    settings = IsolatedSettings(budgets=Budgets(max_tokens_per_run=1_000))
    state = _state({"a": False})
    state["usage"] = Usage(input_tokens=600, output_tokens=400)

    assert route(state, settings=settings) == "stop"


def test_all_passing_assembles_even_over_budget_or_at_the_cap() -> None:
    settings = IsolatedSettings(budgets=Budgets(max_tokens_per_run=1_000))

    assert route(_state({"a": True}, tokens=9_999), settings=settings) == "assemble"
    assert route(_state({"a": True}, revision_count=2), settings=settings) == "assemble"


def test_a_variant_without_a_critique_counts_as_failing() -> None:
    state = _state({"a": True})
    state["variants"].append(_variant("b"))

    assert failing_variant_ids(state) == ["b"]
    assert route(state, settings=IsolatedSettings()) == "revise"


def test_critiques_for_unknown_variants_are_ignored() -> None:
    state = _state({"a": True})
    state["critiques"].append(_critique("ghost", passed=False))

    assert failing_variant_ids(state) == []
    assert route(state, settings=IsolatedSettings()) == "assemble"


def test_no_variants_assembles() -> None:
    state = _state({})

    assert route(state, settings=IsolatedSettings()) == "assemble"


def test_failing_ids_follow_variant_order() -> None:
    state = _state({"c": False, "a": True, "b": False})

    assert failing_variant_ids(state) == ["c", "b"]


def test_route_does_not_change_the_state() -> None:
    state = _state({"a": False, "b": True}, revision_count=1, tokens=10)
    before = {
        "variants": list(state["variants"]),
        "critiques": list(state["critiques"]),
        "revision_count": state["revision_count"],
        "usage": state["usage"],
        "status": state["status"],
    }

    route(state, settings=IsolatedSettings())

    assert {key: state[key] for key in before} == before  # type: ignore[literal-required]


def test_defaults_use_the_documented_limits() -> None:
    # Out of the box: two revisions, 60k tokens. Guards against the defaults drifting.
    settings = IsolatedSettings()

    assert route(_state({"a": False}, revision_count=1), settings=settings) == "revise"
    assert route(_state({"a": False}, revision_count=2), settings=settings) == "stop"
    assert route(_state({"a": False}, tokens=60_000), settings=settings) == "stop"
