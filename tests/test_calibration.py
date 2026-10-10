"""Judge calibration (BF-37). The gateway is faked; these tests make no model call."""

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict
from typer.testing import CliRunner

from brandforge.brands import load_brand
from brandforge.config import Settings
from brandforge.evals.calibration import (
    KAPPA_LOW,
    Agreement,
    CalibrationError,
    CalibrationItem,
    CalibrationReport,
    calibrate,
    format_report,
    load_calibration,
    quadratic_weighted_kappa,
    run_calibration,
)
from brandforge.evals.cases import list_case_ids, load_case
from brandforge.evals.judge import JudgeReply
from brandforge.interfaces import cli
from brandforge.llm.base import GatewayError
from brandforge.models import Usage
from brandforge.scoring import CriterionScore

runner = CliRunner()

EXPECTED_SCORES: dict[str, dict[str, int]] = {
    "brightleaf_01_spring_blossom": {"voice": 5, "clarity": 5, "call_to_action": 5},
    "brightleaf_02_starter_box": {"voice": 1, "clarity": 4, "call_to_action": 2},
    "brightleaf_03_monthly_subscription": {"voice": 3, "clarity": 3, "call_to_action": 3},
    "brightleaf_04_hearth_black": {"voice": 4, "clarity": 4, "call_to_action": 4},
    "brightleaf_05_gift_tin": {"voice": 2, "clarity": 1, "call_to_action": 2},
    "ledgerly_01_vat_reminders": {"voice": 5, "clarity": 5, "call_to_action": 5},
    "ledgerly_02_invoice_chasing": {"voice": 1, "clarity": 4, "call_to_action": 1},
    "ledgerly_03_month_end_close": {"voice": 3, "clarity": 4, "call_to_action": 4},
    "ledgerly_04_receipt_capture": {"voice": 4, "clarity": 5, "call_to_action": 4},
    "ledgerly_05_mileage_log": {"voice": 2, "clarity": 2, "call_to_action": 2},
    "voltride_01_commuter_ebike": {"voice": 5, "clarity": 5, "call_to_action": 5},
    "voltride_02_summit_hill_edition": {"voice": 1, "clarity": 4, "call_to_action": 3},
    "voltride_03_fold_commuter": {"voice": 1, "clarity": 3, "call_to_action": 2},
    "voltride_04_test_ride_weekends": {"voice": 4, "clarity": 4, "call_to_action": 5},
    "voltride_05_lock_and_lights": {"voice": 2, "clarity": 2, "call_to_action": 2},
}

_VALID = """\
id: brightleaf_sample
case_id: brightleaf_01_spring_blossom
channel: email
headline: Spring tea
body: A quiet cup.
cta: Shop the blend
scores:
  voice: 5
  clarity: 4
  call_to_action: 5
note: A sample note.
"""


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def _write(directory: Path, name: str, text: str) -> None:
    directory.joinpath(name).write_text(text, encoding="utf-8")


def _item(
    scores: dict[str, int],
    *,
    headline: str = "Spring tea",
    item_id: str = "brightleaf_gap",
) -> CalibrationItem:
    return CalibrationItem(
        id=item_id,
        case_id="brightleaf_01_spring_blossom",
        brand_id="brightleaf",
        brand_version="1.0",
        rubric_version="1.0",
        channel="email",
        headline=headline,
        body="A quiet cup.",
        cta="Shop the blend",
        scores=scores,
        note="A note that must stay out of the prompt.",
    )


def _reply(scores: dict[str, int]) -> JudgeReply:
    return JudgeReply(
        scores=[CriterionScore(criterion=name, score=score) for name, score in scores.items()]
    )


def test_quadratic_kappa_on_a_one_point_gap_is_one_half() -> None:
    """Human 1, 1, 2, 2 against judge 1, 2, 2, 2. On the fixed 1-5 scale this is 0.5."""
    assert quadratic_weighted_kappa([(1, 1), (1, 2), (2, 2), (2, 2)]) == pytest.approx(0.5)


def test_quadratic_weights_keep_a_two_point_gap_smaller_than_a_full_miss() -> None:
    """Pairs (1, 1), (1, 3), (5, 5).

    Observed disagreement is the weight of a two-point gap, 4/16.
    Expected disagreement on the 1-5 scale is 1.25. Kappa is 1 - 0.25/1.25 = 0.8.
    Linear weights on the same pairs give 2/3, so this pins the quadratic scale.
    """
    assert quadratic_weighted_kappa([(1, 1), (1, 3), (5, 5)]) == pytest.approx(0.8)


def test_perfect_agreement_across_the_scale_is_one() -> None:
    pairs = [(score, score) for score in (1, 2, 3, 4, 5)]
    assert quadratic_weighted_kappa(pairs) == 1.0


def test_a_single_level_makes_kappa_undefined() -> None:
    assert quadratic_weighted_kappa([(5, 5), (5, 5)]) is None
    assert quadratic_weighted_kappa([]) is None


def test_reversed_scores_are_worse_than_chance() -> None:
    assert quadratic_weighted_kappa([(1, 5), (1, 5), (5, 1), (5, 1)]) == -1.0


def test_a_score_outside_the_rubric_is_rejected() -> None:
    with pytest.raises(ValueError, match="1 to 5"):
        quadratic_weighted_kappa([(0, 1)])


def test_the_sample_is_the_first_five_briefs_of_each_brand() -> None:
    items = load_calibration()
    assert [item.id for item in items] == sorted(EXPECTED_SCORES)
    assert {item.id: item.scores for item in items} == EXPECTED_SCORES
    counts = Counter(item.brand_id for item in items)
    assert counts == {"brightleaf": 5, "ledgerly": 5, "voltride": 5}
    for brand_id in counts:
        got = [item.case_id for item in items if item.brand_id == brand_id]
        expected = [case_id for case_id in list_case_ids() if case_id.startswith(f"{brand_id}_")]
        assert got == expected[:5]
        assert all(item.id == item.case_id for item in items if item.brand_id == brand_id)


def test_every_criterion_uses_the_whole_scale() -> None:
    seen: dict[str, set[int]] = {}
    for scores in EXPECTED_SCORES.values():
        for name, score in scores.items():
            seen.setdefault(name, set()).add(score)
    assert seen == {
        "voice": {1, 2, 3, 4, 5},
        "clarity": {1, 2, 3, 4, 5},
        "call_to_action": {1, 2, 3, 4, 5},
    }


def test_each_output_is_a_legal_answer_to_its_brief() -> None:
    for item in load_calibration():
        case = load_case(item.case_id)
        brand = load_brand(item.brand_id)
        assert item.channel in case.brief.channels
        assert item.brand_version == brand.version
        assert item.rubric_version == brand.rubric.version
        cap = case.hard_constraints.headline_max_chars.get(item.channel)
        if cap is not None:
            assert len(item.headline) <= cap
        blob = "\n".join((item.headline, item.body, item.cta)).casefold()
        for phrase in case.hard_constraints.must_mention:
            assert phrase.casefold() in blob
        assert item.note


def test_the_row_hides_the_hand_score() -> None:
    for item in load_calibration():
        payload = json.loads(item.row_json())
        assert payload["system"] == "baseline"
        assert payload["case_id"] == item.case_id
        assert len(payload["variants"]) == 1
        variant = payload["variants"][0]
        assert variant["flagged"] is None
        assert variant["headline"] == item.headline
        assert "note" not in payload
        assert "scores" not in payload
        assert item.note not in item.row_json()


def test_a_judge_that_matches_the_hand_scores_agrees_perfectly() -> None:
    items = load_calibration()
    seen: list[str] = []

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        /,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        assert schema is JudgeReply
        assert tier == "judge"
        assert system is not None
        matches = [item for item in items if f"Headline: {item.headline}\n" in prompt]
        assert len(matches) == 1
        item = matches[0]
        assert item.note not in prompt
        assert item.note not in system
        seen.append(item.id)
        return _reply(item.scores), Usage(input_tokens=2, output_tokens=1)

    report = run_calibration(settings=IsolatedSettings(), complete=fake)

    assert seen == [item.id for item in items]
    assert report.outputs == 15
    assert report.scored == 15
    assert report.unscored == ()
    assert report.overall.count == 45
    assert report.overall.kappa == 1.0
    assert report.overall.exact == 45
    assert report.overall.within_one == 45
    assert report.overall.mae == 0.0
    assert report.row_mean_mae == 0.0
    assert report.disagreements == ()
    assert report.low is False
    assert report.usage.total_tokens == 45
    text = format_report(report)
    assert "The rubric can stand." in text
    assert "Quadratic weighted kappa: 1.00" in text
    assert "(none)" in text


def test_a_judge_stuck_on_one_is_low_agreement() -> None:
    items = load_calibration()

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        /,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        item = next(item for item in items if f"Headline: {item.headline}\n" in prompt)
        return _reply(dict.fromkeys(item.scores, 1)), Usage()

    report = calibrate(items, settings=IsolatedSettings(), complete=fake)

    assert report.overall.kappa is not None
    assert report.overall.kappa < KAPPA_LOW
    assert report.low is True
    assert any(gap.human == 5 and gap.judge == 1 for gap in report.disagreements)
    assert "Tune the rubric" in format_report(report)


def test_a_one_point_shift_is_counted_and_left_off_the_gap_list() -> None:
    item = _item({"voice": 5, "clarity": 4, "call_to_action": 3})

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        /,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        assert "A note that must stay out of the prompt." not in prompt
        return _reply({"voice": 4, "clarity": 5, "call_to_action": 2}), Usage()

    report = calibrate([item], settings=IsolatedSettings(), complete=fake)

    assert report.overall.exact == 0
    assert report.overall.within_one == 3
    assert report.overall.mae == pytest.approx(1.0)
    assert report.disagreements == ()
    assert "(none)" in format_report(report)


def test_a_two_point_gap_is_listed() -> None:
    item = _item({"voice": 5, "clarity": 5, "call_to_action": 5})

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        /,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        return _reply({"voice": 3, "clarity": 5, "call_to_action": 5}), Usage()

    report = calibrate([item], settings=IsolatedSettings(), complete=fake)

    assert len(report.disagreements) == 1
    gap = report.disagreements[0]
    assert gap.criterion == "voice"
    assert gap.human == 5
    assert gap.judge == 3
    assert f"{item.id}  voice  human 5  judge 3" in format_report(report)


def test_a_judge_error_drops_that_output_from_the_kappa() -> None:
    items = load_calibration()
    failed = items[0]

    def fake(
        prompt: str,
        schema: type[JudgeReply],
        tier: str,
        /,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[JudgeReply, Usage]:
        if f"Headline: {failed.headline}\n" in prompt:
            raise GatewayError("judge down")
        item = next(item for item in items if f"Headline: {item.headline}\n" in prompt)
        return _reply(item.scores), Usage()

    report = calibrate(items, settings=IsolatedSettings(), complete=fake)

    assert report.scored == 14
    assert report.overall.count == 42
    assert report.unscored[0][0] == failed.id
    assert "judge down" in report.unscored[0][1]
    assert report.overall.kappa == 1.0
    assert "Unscored" in format_report(report)


def test_an_empty_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="no calibration outputs"):
        load_calibration(tmp_path)


def test_a_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="not found"):
        load_calibration(tmp_path / "missing")


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "bad.yaml", "id: [unclosed")
    with pytest.raises(CalibrationError, match="invalid YAML"):
        load_calibration(tmp_path)


def test_a_non_mapping_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "brightleaf_sample.yaml", "- just a list\n")
    with pytest.raises(CalibrationError, match="YAML mapping"):
        load_calibration(tmp_path)


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "brightleaf_sample.yaml", _VALID + "comment: extra\n")
    with pytest.raises(CalibrationError):
        load_calibration(tmp_path)


def test_the_id_must_match_the_filename(tmp_path: Path) -> None:
    _write(tmp_path, "other.yaml", _VALID)
    with pytest.raises(CalibrationError, match="must match the filename"):
        load_calibration(tmp_path)


def test_the_id_must_start_with_the_brand(tmp_path: Path) -> None:
    text = _VALID.replace("id: brightleaf_sample", "id: ledgerly_sample")
    _write(tmp_path, "ledgerly_sample.yaml", text)
    with pytest.raises(CalibrationError, match="must start with the brand id"):
        load_calibration(tmp_path)


def test_a_channel_the_brief_does_not_request_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "brightleaf_sample.yaml", _VALID.replace("channel: email", "channel: display"))
    with pytest.raises(CalibrationError, match="is not in this brief"):
        load_calibration(tmp_path)


def test_an_unknown_case_is_rejected(tmp_path: Path) -> None:
    text = _VALID.replace("brightleaf_01_spring_blossom", "brightleaf_missing")
    _write(tmp_path, "brightleaf_sample.yaml", text)
    with pytest.raises(CalibrationError, match="Unknown case"):
        load_calibration(tmp_path)


def test_a_score_outside_1_to_5_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "brightleaf_sample.yaml", _VALID.replace("voice: 5", "voice: 6"))
    with pytest.raises(CalibrationError):
        load_calibration(tmp_path)


def test_a_missing_criterion_is_rejected(tmp_path: Path) -> None:
    text = _VALID.replace("  voice: 5\n", "")
    _write(tmp_path, "brightleaf_sample.yaml", text)
    with pytest.raises(CalibrationError, match="scores must be"):
        load_calibration(tmp_path)


def test_a_valid_file_keeps_rubric_order(tmp_path: Path) -> None:
    swapped = _VALID.replace(
        "  voice: 5\n  clarity: 4\n  call_to_action: 5\n",
        "  call_to_action: 5\n  voice: 5\n  clarity: 4\n",
    )
    _write(tmp_path, "brightleaf_sample.yaml", swapped)
    item = load_calibration(tmp_path)[0]
    assert list(item.scores) == ["voice", "clarity", "call_to_action"]
    assert item.scores["clarity"] == 4


def test_cli_prints_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = Agreement(3, 1.0, 3, 3, 0.0, 0.0)
    report = CalibrationReport(
        outputs=1,
        scored=1,
        unscored=(),
        overall=stats,
        by_criterion={"voice": stats},
        by_brand={"brightleaf": stats},
        disagreements=(),
        row_mean_mae=0.0,
        row_mean_signed_error=0.0,
        model="anthropic:claude-opus-5-5",
        prompt_version="v1",
        usage=Usage(),
    )
    monkeypatch.setattr(cli, "run_calibration", lambda: report)

    result = runner.invoke(cli.app, ["calibrate"])

    assert result.exit_code == 0
    assert "The rubric can stand." in result.output
    assert "anthropic:claude-opus-5-5" in result.output


def test_cli_fails_when_the_sample_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> CalibrationReport:
        raise CalibrationError("no calibration outputs in evals/calibration")

    monkeypatch.setattr(cli, "run_calibration", boom)
    result = runner.invoke(cli.app, ["calibrate"])
    assert result.exit_code == 1
    assert "no calibration outputs" in result.output


def test_cli_fails_when_the_judge_scores_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = Agreement(0, None, 0, 0, 0.0, 0.0)
    report = CalibrationReport(
        outputs=1,
        scored=0,
        unscored=(("brightleaf_01_spring_blossom", "judge down"),),
        overall=stats,
        by_criterion={},
        by_brand={},
        disagreements=(),
        row_mean_mae=0.0,
        row_mean_signed_error=0.0,
        model="anthropic:claude-opus-5-5",
        prompt_version="v1",
        usage=Usage(),
    )
    monkeypatch.setattr(cli, "run_calibration", lambda: report)

    result = runner.invoke(cli.app, ["calibrate"])

    assert result.exit_code == 1
    assert "scored nothing" in result.output
    assert "judge down" in result.output


def test_help_lists_calibrate() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "calibrate" in result.output
