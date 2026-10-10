"""Judge rubric (BF-36). The gateway is faked; these tests make no model call."""

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from brandforge.brands import load_brand
from brandforge.config import ModelRef, Settings
from brandforge.evals.cases import load_case
from brandforge.evals.judge import (
    JudgeReply,
    anchored_rubric,
    call_judge,
    crosscheck_settings,
    judge_assertions,
    judge_prompt,
    rubric_block,
    score_row,
)
from brandforge.evals.promptfoo import EvalOutput, EvalUsage, EvalVariant
from brandforge.llm.base import GatewayError, StructuredOutputError
from brandforge.llm.schema import schema_problems
from brandforge.models import Usage, Variant
from brandforge.prompts.loader import load_prompt
from brandforge.scoring import CriterionScore

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "brightleaf_01_spring_blossom"


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def _usage(input_tokens: int = 10, output_tokens: int = 4, cost_usd: float = 0.01) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)


def _variant(variant_id: str, headline: str, channel: str = "email") -> EvalVariant:
    return EvalVariant(
        id=variant_id,
        channel=channel,  # type: ignore[arg-type]
        headline=headline,
        body=f"Body for {headline}",
        cta="Shop the blend",
        flagged=None,
    )


def _row(*variants: EvalVariant, brand_id: str = "brightleaf", rubric_version: str = "1.0") -> str:
    output = EvalOutput(
        system="baseline",
        case_id=CASE_ID,
        brand_id=brand_id,
        brand_version="1.0",
        rubric_version=rubric_version,
        status="complete",
        revision_count=0,
        variants=list(variants),
        errors=[],
        usage=EvalUsage(
            input_tokens=1,
            output_tokens=1,
            cache_write_tokens=0,
            cache_read_tokens=0,
            total_tokens=2,
            cost_usd=0.0,
        ),
    )
    return output.model_dump_json()


def _reply(score: int) -> JudgeReply:
    brand = load_brand("brightleaf")
    return JudgeReply(
        scores=[CriterionScore(criterion=item.name, score=score) for item in brand.rubric.criteria]
    )


def test_the_judge_schema_is_portable() -> None:
    assert schema_problems(JudgeReply) == []


def test_the_prompt_scores_one_variant_against_every_anchor() -> None:
    brand = load_brand("brightleaf")
    case = load_case(CASE_ID)
    shown = Variant(
        id="v1", channel="email", headline="Spring tea", body="A calm cup.", cta="Shop the blend"
    )
    other = Variant(id="v2", channel="social", headline="Other headline", body="x", cta="Look")
    parts = judge_prompt(case.brief, brand, shown, version="v1")

    assert parts.system is not None
    assert "Spring tea" not in parts.system
    assert other.headline not in parts.text
    assert "Spring tea" in parts.user
    assert "<variant>" in parts.user
    assert case.brief.product in parts.user
    for criterion in brand.rubric.criteria:
        assert criterion.name in parts.system
        for anchor in criterion.anchors.values():
            assert anchor in parts.system
            assert anchor in anchored_rubric(brand)
    assert "fixes:" not in load_prompt("judge", "v1")
    assert rubric_block(brand) in anchored_rubric(brand)


def test_each_variant_is_scored_alone_and_the_mean_is_the_row_score() -> None:
    seen: list[tuple[str, str | None, str]] = []

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        assert schema is JudgeReply
        assert tier == "judge"
        assert settings is not None
        seen.append((prompt, system, settings.models.judge))
        score = 5 if "Spring tea" in prompt else 1
        return _reply(score), _usage()

    grade = score_row(
        _row(
            _variant("v1", "Spring tea"),
            _variant("v2", "Other headline", channel="social"),
        ),
        settings=IsolatedSettings(),
        complete=fake,
    )

    assert grade.passed is True
    assert grade.score == pytest.approx(3.0)
    assert grade.criteria == {
        "voice": pytest.approx(3.0),
        "clarity": pytest.approx(3.0),
        "call_to_action": pytest.approx(3.0),
    }
    assert grade.reason.splitlines()[0] == (
        "mean 3.00 (voice 3.00, clarity 3.00, call_to_action 3.00)"
    )
    assert "v1 email: voice 5, clarity 5, call_to_action 5" in grade.reason
    assert "v2 social: voice 1, clarity 1, call_to_action 1" in grade.reason
    assert [item.variant_id for item in grade.variants] == ["v1", "v2"]
    assert len(seen) == 2
    assert "Spring tea" in seen[0][0]
    assert "Other headline" not in seen[0][0]
    assert "Other headline" in seen[1][0]
    assert "Spring tea" not in seen[1][0]
    assert seen[0][1] is not None and "Spring tea" not in seen[0][1]
    assert seen[0][2] == "anthropic:claude-opus-5-5"
    assert grade.usage.input_tokens == 20
    assert grade.usage.cost_usd == pytest.approx(0.02)
    assert grade.model == "anthropic:claude-opus-5-5"


def test_a_low_score_is_still_a_finished_grade() -> None:
    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        return _reply(1), _usage()

    grade = score_row(
        _row(_variant("v1", "Spring tea")), settings=IsolatedSettings(), complete=fake
    )

    assert grade.passed is True
    assert grade.score == pytest.approx(1.0)


def test_a_rubric_version_mismatch_is_named_and_the_disk_rubric_is_used() -> None:
    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        return _reply(4), _usage()

    grade = score_row(
        _row(_variant("v1", "Spring tea"), rubric_version="9.9"),
        settings=IsolatedSettings(),
        complete=fake,
    )

    assert grade.passed is True
    assert grade.rubric_version == "1.0"
    assert "rubric 9.9" in grade.reason
    assert "rubric 1.0" in grade.reason


def test_unusable_rows_make_no_model_call() -> None:
    def boom(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        raise AssertionError("the judge was called")

    settings = IsolatedSettings()
    for output in (
        "not json",
        _row(),
        _row(_variant("v1", "Spring tea"), brand_id="ledgerly"),
        json.dumps({"case_id": "missing_case"}),
    ):
        grade = score_row(output, settings=settings, complete=boom)
        assert grade.passed is False
        assert grade.score == 0.0
        assert grade.reason


def test_a_gateway_error_fails_the_grade_and_keeps_what_was_spent() -> None:
    calls = {"n": 0}

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        calls["n"] += 1
        if calls["n"] == 1:
            return _reply(4), _usage()
        raise StructuredOutputError(
            "nope",
            raw_text="{}",
            usage=_usage(3, 1, 0.002),
            outcome="complete",
        )

    grade = score_row(
        _row(_variant("v1", "Spring tea"), _variant("v2", "Other headline", channel="social")),
        settings=IsolatedSettings(),
        complete=fake,
    )

    assert grade.passed is False
    assert grade.score == 0.0
    assert grade.variants == ()
    assert "v2:" in grade.reason
    assert grade.usage.input_tokens == 13
    assert grade.usage.output_tokens == 5
    assert grade.usage.cost_usd == pytest.approx(0.012)


def test_a_plain_gateway_error_fails_that_variant() -> None:
    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        raise GatewayError("judge down")

    grade = score_row(
        _row(_variant("v1", "Spring tea")), settings=IsolatedSettings(), complete=fake
    )

    assert grade.passed is False
    assert "GatewayError: judge down" in grade.reason
    assert grade.usage.total_tokens == 0


def test_a_reply_that_misses_the_rubric_fails_the_grade() -> None:
    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        return JudgeReply(scores=[CriterionScore(criterion="voice", score=9)]), _usage()

    grade = score_row(
        _row(_variant("v1", "Spring tea")), settings=IsolatedSettings(), complete=fake
    )

    assert grade.passed is False
    assert grade.usage.input_tokens == 10
    assert "voice" in grade.reason


def test_assertions_use_the_judge_and_add_a_crosscheck_when_configured() -> None:
    brand = load_brand("brightleaf")
    alone = judge_assertions(brand, settings=IsolatedSettings())
    assert [item["metric"] for item in alone] == ["judge"]
    assert alone[0]["type"] == "llm-rubric"
    assert alone[0]["weight"] == 0
    assert alone[0]["rubricPrompt"] == "{{output}}"
    assert alone[0]["provider"]["config"] == {"role": "judge"}

    both = judge_assertions(
        brand,
        settings=IsolatedSettings(
            judge_crosscheck_model="gemini:gemini-test",
            judge_crosscheck_input_per_mtok=0.5,
            judge_crosscheck_output_per_mtok=3.0,
        ),
    )
    assert [item["metric"] for item in both] == ["judge", "judge_crosscheck"]
    assert both[1]["provider"]["config"] == {"role": "crosscheck"}
    assert both[0]["value"] == both[1]["value"]


def test_crosscheck_settings_borrow_the_judge_tier_without_the_opus_cache_rate() -> None:
    original = IsolatedSettings(
        judge_crosscheck_model="gemini:gemini-test",
        judge_crosscheck_input_per_mtok=0.5,
        judge_crosscheck_output_per_mtok=3.0,
    )
    copied = crosscheck_settings(original)

    assert copied.models.resolve("judge") == ModelRef("gemini", "gemini-test")
    assert copied.pricing.judge.input_per_mtok == 0.5
    assert copied.pricing.judge.output_per_mtok == 3.0
    assert copied.pricing.judge.cache_read_multiplier is None
    assert original.models.judge == "anthropic:claude-opus-5-5"
    assert original.pricing.judge.cache_read_multiplier == 0.05
    with pytest.raises(ValueError, match="cross-check is off"):
        crosscheck_settings(IsolatedSettings())


def test_call_judge_returns_the_promptfoo_grade_and_the_judge_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system
        assert settings is not None
        assert settings.models.judge == "anthropic:claude-opus-5-5"
        return _reply(4), _usage()

    monkeypatch.setattr("brandforge.evals.judge.complete_structured", fake)
    monkeypatch.setattr("brandforge.evals.judge.get_settings", IsolatedSettings)
    result = call_judge(_row(_variant("v1", "Spring tea")), {}, None)

    assert result["output"]["pass"] is True
    assert result["output"]["score"] == pytest.approx(4.0)
    assert result["tokenUsage"] == {"prompt": 10, "completion": 4, "total": 14}
    assert result["cost"] == pytest.approx(0.01)
    assert result["metadata"]["criteria"]["voice"] == pytest.approx(4.0)
    assert result["metadata"]["variants"][0]["id"] == "v1"


def test_call_judge_reads_a_wrapped_prompt_from_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del schema, tier, system, settings
        assert "Spring tea" in prompt
        return _reply(4), _usage()

    monkeypatch.setattr("brandforge.evals.judge.complete_structured", fake)
    monkeypatch.setattr("brandforge.evals.judge.get_settings", IsolatedSettings)
    row = _row(_variant("v1", "Spring tea"))
    result = call_judge("Grade this row", {}, {"vars": {"output": row}})

    assert result["output"]["pass"] is True


def test_a_crosscheck_call_uses_the_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system
        assert settings is not None
        seen.append(settings.models.judge)
        return _reply(4), _usage()

    monkeypatch.setattr("brandforge.evals.judge.complete_structured", fake)
    monkeypatch.setattr(
        "brandforge.evals.judge.get_settings",
        lambda: IsolatedSettings(
            judge_crosscheck_model="gemini:gemini-test",
            judge_crosscheck_input_per_mtok=0.5,
            judge_crosscheck_output_per_mtok=3.0,
        ),
    )
    result = call_judge(
        _row(_variant("v1", "Spring tea")),
        {"config": {"role": "crosscheck"}},
        None,
    )

    assert result["output"]["pass"] is True
    assert seen == ["gemini:gemini-test"]
    assert result["metadata"]["role"] == "crosscheck"
    assert result["metadata"]["judge_model"] == "gemini:gemini-test"


def test_a_crosscheck_with_no_model_does_not_call_the_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        raise AssertionError("the judge was called")

    monkeypatch.setattr("brandforge.evals.judge.complete_structured", boom)
    monkeypatch.setattr("brandforge.evals.judge.get_settings", IsolatedSettings)
    result = call_judge("{}", {"config": {"role": "crosscheck"}}, None)

    assert result["output"]["pass"] is False
    assert "cross-check" in result["output"]["reason"]


def test_an_unknown_role_fails_before_a_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        del prompt, schema, tier, system, settings
        raise AssertionError("the judge was called")

    monkeypatch.setattr("brandforge.evals.judge.complete_structured", boom)
    result = call_judge("{}", {"config": {"role": "writer"}}, None)

    assert result["output"]["pass"] is False
    assert "role" in result["output"]["reason"]


def test_the_provider_script_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[object, Mapping[str, Any] | None]] = []

    def fake(
        prompt: object,
        options: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del context
        seen.append((prompt, options))
        return {"output": {"pass": True, "score": 4.0, "reason": "ok"}}

    monkeypatch.setattr("brandforge.evals.judge.call_judge", fake)
    script = _load_script(ROOT / "evals" / "providers" / "judge.py")
    assert script.call_api("row", {"config": {"role": "judge"}}, None)["output"]["pass"] is True
    assert seen == [("row", {"config": {"role": "judge"}})]


def _load_script(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"promptfoo_{path.stem}_judge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
