"""`build_critique`: the conversion from the model's reply to a `Critique`, no model involved."""

import pytest

from brandforge.brands.loader import load_brand
from brandforge.config import Thresholds
from brandforge.llm.schema import schema_problems
from brandforge.models import Rubric
from brandforge.scoring import CriterionScore, CriticReply, CritiqueError, build_critique

DEFAULT = Thresholds()  # overall >= 4.0, no criterion below 3


@pytest.fixture
def rubric() -> Rubric:
    return load_brand("voltride").rubric  # criteria: voice, clarity, call_to_action


def _reply(
    voice: int = 5,
    clarity: int = 5,
    call_to_action: int = 5,
    fixes: list[str] | None = None,
) -> CriticReply:
    return CriticReply(
        scores=[
            CriterionScore(criterion="voice", score=voice),
            CriterionScore(criterion="clarity", score=clarity),
            CriterionScore(criterion="call_to_action", score=call_to_action),
        ],
        fixes=fixes or [],
    )


def test_scores_follow_the_rubric_order_whatever_order_the_model_used(rubric: Rubric) -> None:
    reply = CriticReply(
        scores=[
            CriterionScore(criterion="call_to_action", score=3),
            CriterionScore(criterion="voice", score=5),
            CriterionScore(criterion="clarity", score=4),
        ],
        fixes=[],
    )

    critique = build_critique("search-1", reply, rubric, DEFAULT)

    assert list(critique.scores.items()) == [("voice", 5), ("clarity", 4), ("call_to_action", 3)]


def test_variant_id_and_fixes_are_carried_through(rubric: Rubric) -> None:
    reply = _reply(3, 5, 5, fixes=["Add a concrete number to the headline"])

    critique = build_critique("social-2", reply, rubric, DEFAULT)

    assert critique.variant_id == "social-2"
    assert critique.fixes == ["Add a concrete number to the headline"]


def test_overall_is_the_mean_of_the_criterion_scores(rubric: Rubric) -> None:
    critique = build_critique("search-1", _reply(5, 4, 3), rubric, DEFAULT)

    assert critique.overall == pytest.approx(4.0)


def test_all_fives_pass(rubric: Rubric) -> None:
    assert build_critique("search-1", _reply(), rubric, DEFAULT).passed is True


def test_overall_exactly_at_the_threshold_passes(rubric: Rubric) -> None:
    # Mean 4.0, lowest score 3: both rules are met on their boundaries.
    assert build_critique("search-1", _reply(5, 4, 3), rubric, DEFAULT).passed is True


def test_overall_below_the_threshold_fails(rubric: Rubric) -> None:
    critique = build_critique("search-1", _reply(4, 4, 3), rubric, DEFAULT)

    assert critique.overall == pytest.approx(11 / 3)
    assert critique.passed is False


def test_one_criterion_below_the_floor_fails_even_when_the_mean_is_high(rubric: Rubric) -> None:
    critique = build_critique("search-1", _reply(5, 5, 2), rubric, DEFAULT)

    assert critique.overall == pytest.approx(4.0)
    assert critique.passed is False


def test_thresholds_come_from_config_not_from_the_model(rubric: Rubric) -> None:
    reply = _reply(5, 4, 4)  # mean 4.33

    assert build_critique("a", reply, rubric, Thresholds(min_overall=4.0)).passed is True
    assert build_critique("a", reply, rubric, Thresholds(min_overall=4.5)).passed is False
    assert build_critique("a", reply, rubric, Thresholds(min_per_criterion=5)).passed is False


def test_a_missing_criterion_is_rejected(rubric: Rubric) -> None:
    reply = CriticReply(
        scores=[
            CriterionScore(criterion="voice", score=5),
            CriterionScore(criterion="clarity", score=5),
        ],
        fixes=[],
    )

    with pytest.raises(CritiqueError, match="missing criteria: call_to_action"):
        build_critique("search-1", reply, rubric, DEFAULT)


def test_an_unknown_criterion_is_rejected(rubric: Rubric) -> None:
    reply = _reply()
    reply.scores.append(CriterionScore(criterion="humour", score=5))

    with pytest.raises(CritiqueError, match="unknown criteria: humour"):
        build_critique("search-1", reply, rubric, DEFAULT)


def test_a_criterion_scored_twice_is_rejected(rubric: Rubric) -> None:
    reply = _reply()
    reply.scores.append(CriterionScore(criterion="voice", score=1))

    with pytest.raises(CritiqueError, match="more than once: voice"):
        build_critique("search-1", reply, rubric, DEFAULT)


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_a_score_outside_one_to_five_is_rejected(rubric: Rubric, bad: int) -> None:
    with pytest.raises(CritiqueError, match=rf"scores outside 1 to 5: clarity={bad}"):
        build_critique("search-1", _reply(clarity=bad), rubric, DEFAULT)


def test_the_error_names_the_variant(rubric: Rubric) -> None:
    with pytest.raises(CritiqueError, match="'social-3'"):
        build_critique("social-3", _reply(clarity=9), rubric, DEFAULT)


def test_reply_schema_is_portable() -> None:
    assert schema_problems(CriticReply) == []
