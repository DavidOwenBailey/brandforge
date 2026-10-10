"""promptfoo providers for the baseline and the pipeline. No model calls."""

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from brandforge.brands import load_brand
from brandforge.evals import (
    EvalOutput,
    call_baseline,
    call_pipeline,
    generate_tests,
    list_case_ids,
    load_case,
)
from brandforge.models import RunError, RunResult, Usage, Variant, VariantResult, new_run_state

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
CASE_ID = "brightleaf_01_spring_blossom"


def _variant(variant_id: str = "v1") -> Variant:
    return Variant(
        id=variant_id,
        channel="email",
        headline="Spring tea",
        body="A calm cup.",
        cta="Shop",
    )


def _usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=4, cache_read_tokens=2, cost_usd=0.25)


def _parse(response: Mapping[str, Any]) -> EvalOutput:
    assert "error" not in response
    output = response["output"]
    assert isinstance(output, str)
    return EvalOutput.model_validate_json(output)


def test_generate_tests_lists_every_case_and_the_judge_rubric() -> None:
    from brandforge.config import get_settings

    tests = generate_tests()
    assert [test["description"] for test in tests] == list_case_ids()
    assert all(test["vars"] == {"case_id": test["description"]} for test in tests)
    blossom = next(test for test in tests if test["description"] == CASE_ID)
    assert blossom["metadata"] == {"brand_id": "brightleaf"}
    judge = blossom["assert"][0]
    assert judge["type"] == "llm-rubric"
    assert judge["metric"] == "judge"
    assert "Brightleaf" in judge["value"]
    assert "\n  5:" in judge["value"]
    ledger = next(test for test in tests if test["description"] == "ledgerly_01_vat_reminders")
    assert "Ledgerly" in ledger["assert"][0]["value"]
    metrics = [item["metric"] for item in blossom["assert"]]
    if get_settings().judge_crosscheck is None:
        assert metrics == ["judge"]
    else:
        assert metrics == ["judge", "judge_crosscheck"]


def test_generate_tests_keeps_a_requested_subset_in_the_given_order() -> None:
    wanted = ["voltride_01_commuter_ebike", "brightleaf_01_spring_blossom"]
    tests = generate_tests({"case_ids": [f"  {wanted[0]}", wanted[1]]})
    assert [test["description"] for test in tests] == wanted


def test_generate_tests_rejects_an_unknown_or_blank_id() -> None:
    with pytest.raises(ValueError, match="Unknown case"):
        generate_tests({"case_ids": [CASE_ID, "missing_case"]})
    with pytest.raises(ValueError, match="case_ids must be a list"):
        generate_tests({"case_ids": "brightleaf_01_spring_blossom"})


def test_an_empty_id_list_runs_nothing() -> None:
    assert generate_tests({"case_ids": []}) == []


def test_baseline_returns_the_shared_row_and_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_baseline(brief: Any, brand: Any, **kwargs: Any) -> tuple[list[Variant], Usage]:
        seen["brief"] = brief
        seen["brand_id"] = brand.id
        seen["kwargs"] = kwargs
        return [_variant()], _usage()

    monkeypatch.setattr("brandforge.baseline.generate_baseline", fake_baseline)
    case = load_case(CASE_ID)
    response = call_baseline("ignored", {}, {"vars": {"case_id": f"  {CASE_ID}  "}})

    row = _parse(response)
    brand = load_brand("brightleaf")
    assert seen["brief"] == case.brief
    assert seen["brand_id"] == "brightleaf"
    assert seen["kwargs"] == {}
    assert row.system == "baseline"
    assert row.case_id == CASE_ID
    assert row.brand_id == brand.id
    assert row.brand_version == brand.version
    assert row.rubric_version == brand.rubric.version
    assert row.status == "complete"
    assert row.run_id is None
    assert row.trace_id is None
    assert row.revision_count == 0
    assert row.errors == []
    assert row.variants[0].flagged is None
    assert row.variants[0].headline == "Spring tea"
    assert row.usage.total_tokens == 16
    assert response["tokenUsage"] == {"total": 16, "prompt": 12, "completion": 4, "numRequests": 1}
    assert response["cost"] == 0.25
    assert json.loads(response["output"])["variants"][0]["flagged"] is None


def test_pipeline_copies_the_run_result_and_omits_the_request_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(brief: Any, brand: Any, **kwargs: Any) -> Any:
        assert kwargs == {}
        state = new_run_state(brief, brand, run_id="run-1", trace_id="trace-1")
        state["result"] = RunResult(
            run_id="run-1",
            status="partial",
            brand_id=brand.id,
            brand_version=brand.version,
            rubric_version=brand.rubric.version,
            variants=[VariantResult(variant=_variant("pipe-1"), critique=None, flagged=True)],
            revision_count=2,
            errors=[RunError(node="writer", message="short copy", fatal=False)],
            usage=_usage(),
            trace_id="trace-1",
        )
        return state

    monkeypatch.setattr("brandforge.graph.run_graph", fake_run)
    response = call_pipeline(CASE_ID, {}, {})

    row = _parse(response)
    assert row.system == "pipeline"
    assert row.status == "partial"
    assert row.run_id == "run-1"
    assert row.trace_id == "trace-1"
    assert row.revision_count == 2
    assert row.errors == ["writer: short copy"]
    assert row.variants[0].id == "pipe-1"
    assert row.variants[0].flagged is True
    assert "numRequests" not in response["tokenUsage"]
    assert response["tokenUsage"]["prompt"] == 12


def test_vars_win_over_the_rendered_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_baseline(brief: Any, brand: Any, **kwargs: Any) -> tuple[list[Variant], Usage]:
        del brief, kwargs
        seen.append(brand.id)
        return [_variant()], Usage()

    monkeypatch.setattr("brandforge.baseline.generate_baseline", fake_baseline)
    call_baseline(
        "ledgerly_01_vat_reminders",
        {},
        {"vars": {"case_id": CASE_ID}},
    )
    assert seen == ["brightleaf"]


def test_a_missing_case_is_an_error_row_not_a_raise() -> None:
    response = call_baseline("   ", {}, {})
    assert response == {
        "error": (
            "ValueError: promptfoo test has no case_id. "
            "Set vars.case_id or use the {{case_id}} prompt."
        )
    }
    missing = call_pipeline("no-such-case", {}, {"vars": {"case_id": "no-such-case"}})
    assert missing["error"].startswith("CaseLoadError:")
    assert "output" not in missing


def test_a_system_failure_is_an_error_row(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(brief: Any, brand: Any, **kwargs: Any) -> tuple[list[Variant], Usage]:
        del brief, brand, kwargs
        raise RuntimeError("gateway\ndown")

    monkeypatch.setattr("brandforge.baseline.generate_baseline", boom)
    assert call_baseline(CASE_ID, {}, {}) == {"error": "RuntimeError: gateway down"}


def test_a_pipeline_without_a_result_is_an_error_row(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(brief: Any, brand: Any, **kwargs: Any) -> Any:
        del kwargs
        return new_run_state(brief, brand, run_id="run-1")

    monkeypatch.setattr("brandforge.graph.run_graph", fake_run)
    response = call_pipeline(CASE_ID, {}, {})
    assert response["error"] == "RuntimeError: the pipeline finished without a result"


def test_config_runs_both_providers_over_the_case_generator() -> None:
    raw = yaml.safe_load((EVALS / "promptfooconfig.yaml").read_text(encoding="utf-8"))
    providers = raw["providers"]
    assert raw["prompts"] == ["{{case_id}}"]
    assert raw["tests"] == ["file://providers/tests.py:generate_tests"]
    assert [item["label"] for item in providers] == ["baseline", "pipeline"]
    assert [item["id"] for item in providers] == [
        "file://providers/baseline.py",
        "file://providers/pipeline.py",
    ]
    assert providers[1]["config"]["timeout"] == 600000
    assert raw["commandLineOptions"]["cache"] is False
    assert raw["commandLineOptions"]["maxConcurrency"] == 1
    assert raw["defaultTest"]["assert"] == [
        {"type": "is-json", "metric": "valid_json"},
        {
            "type": "python",
            "value": "file://providers/assertions.py:assert_json",
            "metric": "eval_row",
        },
        {
            "type": "python",
            "value": "file://providers/assertions.py:assert_channel_limits",
            "metric": "channel_limits",
        },
        {
            "type": "python",
            "value": "file://providers/assertions.py:assert_banned_words",
            "metric": "banned_words",
        },
        {
            "type": "python",
            "value": "file://providers/assertions.py:assert_cta",
            "metric": "cta_present",
        },
        {
            "type": "python",
            "value": "file://providers/assertions.py:assert_must_mention",
            "metric": "must_mention",
        },
    ]


def test_provider_scripts_delegate_to_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_baseline(
        prompt: str, options: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, Any]:
        del options, context
        seen.append(f"baseline:{prompt}")
        return {"output": "{}"}

    def fake_pipeline(
        prompt: str, options: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, Any]:
        del options, context
        seen.append(f"pipeline:{prompt}")
        return {"output": "{}"}

    monkeypatch.setattr("brandforge.evals.promptfoo.call_baseline", fake_baseline)
    monkeypatch.setattr("brandforge.evals.promptfoo.call_pipeline", fake_pipeline)

    baseline = _load_script(EVALS / "providers" / "baseline.py")
    pipeline = _load_script(EVALS / "providers" / "pipeline.py")
    tests = _load_script(EVALS / "providers" / "tests.py")

    assert baseline.call_api(CASE_ID, {}, {}) == {"output": "{}"}
    assert pipeline.call_api("other", {}, {}) == {"output": "{}"}
    assert seen == [f"baseline:{CASE_ID}", "pipeline:other"]
    assert tests.generate_tests({"case_ids": [CASE_ID]})[0]["description"] == CASE_ID


def _load_script(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"promptfoo_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
