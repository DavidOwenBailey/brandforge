from typing import Any

import pytest
from pydantic import ValidationError

from brandforge.models import BrandProfile, Brief, Critique, Usage, Variant


def valid_rubric() -> dict[str, Any]:
    anchors = {i: f"level {i}" for i in range(1, 6)}
    return {
        "version": "1.0",
        "criteria": [
            {
                "name": "voice",
                "description": "Fits brand voice",
                "anchors": anchors,
            }
        ],
    }


def valid_brief(**overrides: Any) -> dict[str, Any]:
    data = {
        "product": "Trail shoes",
        "audience": "weekend hikers",
        "objective": "conversion",
        "channels": ["search", "email"],
    }
    return {**data, **overrides}


class TestBrief:
    def test_valid_brief_defaults_constraints(self) -> None:
        assert Brief(**valid_brief()).constraints == []

    @pytest.mark.parametrize(
        "overrides",
        [
            {"channels": []},
            {"channels": ["search", "search"]},
            {"channels": ["tiktok"]},
            {"objective": "virality"},
            {"product": "   "},
            {"constraints": [""]},
        ],
    )
    def test_invalid_briefs_rejected(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Brief(**valid_brief(**overrides))

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Brief(**valid_brief(tone="funny"))

    def test_whitespace_is_stripped(self) -> None:
        assert Brief(**valid_brief(product="  Trail shoes  ")).product == "Trail shoes"


class TestBrandProfile:
    def valid(self, **overrides: Any) -> dict[str, Any]:
        data = {"id": "acme", "version": "1.0", "voice": ["warm"], "rubric": valid_rubric()}
        return {**data, **overrides}

    def test_valid_profile(self) -> None:
        assert BrandProfile(**self.valid()).banned_words == []

    def test_voice_required(self) -> None:
        with pytest.raises(ValidationError):
            BrandProfile(**self.valid(voice=[]))

    def test_rubric_anchors_must_cover_1_to_5(self) -> None:
        rubric = valid_rubric()
        del rubric["criteria"][0]["anchors"][3]
        with pytest.raises(ValidationError):
            BrandProfile(**self.valid(rubric=rubric))

    def test_duplicate_criterion_names_rejected(self) -> None:
        rubric = valid_rubric()
        rubric["criteria"].append(rubric["criteria"][0])
        with pytest.raises(ValidationError):
            BrandProfile(**self.valid(rubric=rubric))

    def test_rubric_version_required(self) -> None:
        rubric = valid_rubric()
        del rubric["version"]
        with pytest.raises(ValidationError):
            BrandProfile(**self.valid(rubric=rubric))


class TestVariant:
    def test_valid_variant(self) -> None:
        v = Variant(id="v1", channel="social", headline="Hi", body="", cta="Shop")
        assert v.channel == "social"

    def test_bad_channel_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Variant.model_validate(
                {"id": "v1", "channel": "radio", "headline": "Hi", "body": "x", "cta": "Shop"}
            )

    def test_empty_headline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Variant(id="v1", channel="social", headline=" ", body="x", cta="Shop")


class TestCritique:
    def test_valid_critique(self) -> None:
        c = Critique(variant_id="v1", scores={"voice": 4}, overall=4.0, passed=True)
        assert c.fixes == []

    @pytest.mark.parametrize("score", [0, 6, -1])
    def test_score_out_of_range_rejected(self, score: int) -> None:
        with pytest.raises(ValidationError):
            Critique(variant_id="v1", scores={"voice": score}, overall=3.0, passed=False)

    def test_overall_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Critique(variant_id="v1", scores={"voice": 4}, overall=5.5, passed=True)

    def test_scores_required(self) -> None:
        with pytest.raises(ValidationError):
            Critique(variant_id="v1", scores={}, overall=3.0, passed=False)


class TestUsage:
    def test_defaults_are_zero(self) -> None:
        assert Usage().total_tokens == 0

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Usage(input_tokens=-1)

    def test_addition_accumulates(self) -> None:
        total = Usage(
            input_tokens=10,
            output_tokens=5,
            cache_write_tokens=3,
            cache_read_tokens=4,
            cost_usd=0.01,
        ) + Usage(input_tokens=1, output_tokens=2, cache_read_tokens=6, cost_usd=0.02)
        assert (total.input_tokens, total.output_tokens) == (11, 7)
        assert (total.cache_write_tokens, total.cache_read_tokens) == (3, 10)
        assert total.total_tokens == 11 + 7 + 3 + 10
        assert total.cost_usd == pytest.approx(0.03)

    def test_a_node_that_runs_twice_is_summed_into_one_row(self) -> None:
        first = Usage(input_tokens=10, output_tokens=1, cost_usd=0.1).attributed_to("critic")
        writer = Usage(
            input_tokens=3, output_tokens=1, cache_read_tokens=8, cost_usd=0.05
        ).attributed_to("writer")
        second = Usage(input_tokens=4, output_tokens=2, cost_usd=0.2).attributed_to("critic")

        total = first + writer + second

        assert [item.node for item in total.nodes] == ["critic", "writer"]
        assert total.nodes[0].input_tokens == 14
        assert total.nodes[0].cost_usd == pytest.approx(0.3)
        assert total.nodes[1].cache_read_tokens == 8
        assert total.nodes[1].total_tokens == 3 + 1 + 8

    def test_attributing_nothing_or_twice_adds_no_row(self) -> None:
        assert Usage().attributed_to("planner").nodes == []
        once = Usage(input_tokens=1, cost_usd=0.01).attributed_to("planner")
        assert [item.node for item in once.attributed_to("planner").nodes] == ["planner"]
