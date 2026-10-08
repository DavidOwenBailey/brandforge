"""CLI tests. The model is faked by replacing the planner, writer, critic and reviser in the
graph module. Checkpoints go to a database in the test's temp folder, never the working one."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict
from typer.testing import CliRunner

from brandforge import __version__, graph, tracing
from brandforge.config import Settings
from brandforge.interfaces import cli
from brandforge.llm.base import GatewayConfigError
from brandforge.models import Critique, Plan, RunError, RunState, Usage, Variant

runner = CliRunner()

BRIEF_YAML = """\
product: Commuter e-bike
audience: city commuters
objective: conversion
channels: [search, social]
constraints:
  - no discounts
"""


def _variant(n: int, channel: str = "search") -> Variant:
    return Variant.model_validate(
        {
            "id": f"{channel}-{n}",
            "channel": channel,
            "headline": f"Headline {n}",
            "body": f"Body {n}",
            "cta": f"CTA {n}",
        }
    )


class FakeGenerate:
    """Stands in for the writer node: reads state, returns a partial state update."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[RunState] = []

    def __call__(self, state: RunState) -> dict[str, Any]:
        self.calls.append(state)
        if self.error:
            raise self.error
        usage = Usage(input_tokens=120, output_tokens=80, cost_usd=0.0012)
        return {"variants": [_variant(1), _variant(2, "social")], "usage": usage}


def _fake_plan(state: RunState) -> dict[str, Any]:
    plan = Plan(
        audience="commuters",
        angle="save time",
        channels=list(state["brief"].channels),
        variants_per_channel=2,
    )
    return {"plan": plan, "usage": Usage()}  # zero usage keeps the printed totals the generator's


def _fake_critique(state: RunState) -> dict[str, Any]:
    # Every variant passes, so the router assembles and the reviser is never needed.
    critiques = [
        Critique(variant_id=v.id, scores={"voice": 5}, overall=5.0, passed=True)
        for v in state["variants"]
    ]
    return {"critiques": critiques, "usage": Usage()}  # zero usage: totals stay the writer's


def _fake_revise(state: RunState) -> dict[str, Any]:
    # A guard: if a CLI test ever needs a revision it must fake one, not reach a real model.
    raise AssertionError("the reviser should not run in CLI tests")


@pytest.fixture(autouse=True)
def fake_planner_critic_and_reviser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "plan_brief", _fake_plan)
    monkeypatch.setattr(graph, "critique_variants", _fake_critique)
    monkeypatch.setattr(graph, "revise_variants", _fake_revise)


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


@pytest.fixture(autouse=True)
def checkpoint_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send the CLI's checkpoints to a temp database instead of `.brandforge/` in the cwd."""
    path = tmp_path / "state" / "checkpoints.sqlite"
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings(checkpoint_db=path))
    return path


@pytest.fixture
def brief_file(tmp_path: Path) -> Path:
    path = tmp_path / "brief.yaml"
    path.write_text(BRIEF_YAML, encoding="utf-8")
    return path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeGenerate:
    fake = FakeGenerate()
    monkeypatch.setattr(graph, "write_variants", fake)
    return fake


def test_generate_prints_variants_and_cost(fake: FakeGenerate, brief_file: Path) -> None:
    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 0
    assert "[search-1] search" in result.stdout
    assert "Headline 2" in result.stdout
    assert "CTA 2" in result.stdout
    assert "Tokens: 120 in, 80 out (200 total)" in result.stdout
    assert "Cost:   $0.0012" in result.stdout
    assert "Cost by node:" in result.stdout
    assert "writer" in result.stdout.split("Cost by node:", 1)[1]


def test_format_usage_lists_cache_tokens_and_one_row_per_node() -> None:
    usage = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_write_tokens=100,
        cost_usd=0.01,
    ).attributed_to("planner") + Usage(
        input_tokens=4, output_tokens=2, cache_read_tokens=80, cost_usd=0.002
    ).attributed_to("critic")

    text = cli.format_usage(usage)

    assert "Tokens: 14 in, 7 out, 100 cache write, 80 cache read (201 total)" in text
    assert "Cost:   $0.0120" in text
    assert "Cost by node:" in text
    assert "Cache write" in text
    assert "planner" in text
    assert "critic" in text


def test_generate_says_tracing_is_off_when_the_run_has_no_trace(
    fake: FakeGenerate, brief_file: Path
) -> None:
    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 0
    assert "Trace ID: (tracing is off)" in result.stdout


def test_generate_prints_the_trace_id(
    fake: FakeGenerate, brief_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracing, "trace_id_for", lambda run_id, settings=None: "abc123")

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 0
    assert "Trace ID: abc123" in result.stdout


def test_generate_prints_a_summary_table_of_scores_and_flags(
    fake: FakeGenerate, brief_file: Path
) -> None:
    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 0
    assert "Status:    complete" in result.stdout
    assert "Brand:     voltride v" in result.stdout
    assert "Variants:  2 (0 flagged)" in result.stdout
    assert "Revisions: 0" in result.stdout
    rows = [line.split() for line in result.stdout.splitlines()]
    assert ["Variant", "Channel", "voice", "Overall", "Result"] in rows
    assert ["search-1", "search", "5", "5.0", "passed"] in rows
    assert ["social-2", "social", "5", "5.0", "passed"] in rows
    assert "Still to fix:" not in result.stdout
    assert "Errors:" not in result.stdout


def _failing_critique(state: RunState) -> dict[str, Any]:
    critiques = [
        Critique(
            variant_id=v.id,
            scores={"voice": 2},
            overall=2.0,
            passed=False,
            fixes=["Tighten the headline"],
        )
        for v in state["variants"]
    ]
    return {"critiques": critiques, "usage": Usage()}


def _counting_revise(state: RunState) -> dict[str, Any]:
    # Changes nothing but the count, so the loop runs to the cap and the router stops it.
    return {"critiques": [], "revision_count": state["revision_count"] + 1, "usage": Usage()}


def test_flagged_variants_are_marked_with_their_remaining_fixes(
    fake: FakeGenerate, brief_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph, "critique_variants", _failing_critique)
    monkeypatch.setattr(graph, "revise_variants", _counting_revise)

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 0
    assert "Status:    partial" in result.stdout
    assert "Variants:  2 (2 flagged)" in result.stdout
    assert "Revisions: 2" in result.stdout
    rows = [line.split() for line in result.stdout.splitlines()]
    assert ["search-1", "search", "2", "2.0", "flagged"] in rows
    assert "Still to fix:" in result.stdout
    assert "[search-1] Tighten the headline" in result.stdout


def test_a_run_with_no_variants_is_reported_failed_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, brief_file: Path
) -> None:
    def empty_writer(state: RunState) -> dict[str, Any]:
        return {"variants": [], "usage": Usage()}

    monkeypatch.setattr(graph, "write_variants", empty_writer)

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 1
    assert "(no variants were produced)" in result.stdout
    assert "Status:    failed" in result.stdout
    assert "Variant  Channel" not in result.stdout  # no rows, so no table


def test_errors_recorded_during_the_run_are_listed(
    monkeypatch: pytest.MonkeyPatch, brief_file: Path
) -> None:
    def short_writer(state: RunState) -> dict[str, Any]:
        return {
            "variants": [_variant(1)],
            "errors": [RunError(node="writer", message="channel 'social': got 0 variants")],
            "usage": Usage(),
        }

    monkeypatch.setattr(graph, "write_variants", short_writer)

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 0
    assert "Status:    partial" in result.stdout
    assert "Errors:" in result.stdout
    assert "[writer] channel 'social': got 0 variants" in result.stdout


def test_generate_passes_validated_brief_and_brand(fake: FakeGenerate, brief_file: Path) -> None:
    runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    state = fake.calls[0]
    brief, brand = state["brief"], state["brand"]
    assert brief.product == "Commuter e-bike"
    assert brief.channels == ["search", "social"]
    assert brand.id == "voltride"


def test_missing_brief_file_is_a_usage_error(fake: FakeGenerate, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app, ["generate", "--brand", "voltride", "--brief", str(tmp_path / "nope.yaml")]
    )

    assert result.exit_code == 2
    assert fake.calls == []


def test_invalid_brief_fails_cleanly(fake: FakeGenerate, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("product: only this\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(bad)])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert fake.calls == []


def test_brief_that_is_not_a_mapping_fails_cleanly(fake: FakeGenerate, tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(bad)])

    assert result.exit_code == 1
    assert "must be a YAML mapping" in result.output


def test_malformed_yaml_fails_cleanly(fake: FakeGenerate, tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("product: [unclosed\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(bad)])

    assert result.exit_code == 1
    assert "cannot read" in result.output


def test_unknown_brand_lists_known_brands(fake: FakeGenerate, brief_file: Path) -> None:
    result = runner.invoke(cli.app, ["generate", "--brand", "nope", "--brief", str(brief_file)])

    assert result.exit_code == 1
    assert "Unknown brand 'nope'" in result.output
    assert "voltride" in result.output
    assert fake.calls == []


def test_a_node_failure_is_reported_in_the_summary_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, brief_file: Path
) -> None:
    monkeypatch.setattr(
        graph,
        "write_variants",
        FakeGenerate(GatewayConfigError("ANTHROPIC_API_KEY is not set.")),
    )

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 1
    assert "Status:    failed" in result.output
    assert "Errors:" in result.output
    assert "[writer] GatewayConfigError: ANTHROPIC_API_KEY is not set." in result.output


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(cli.app, [])

    assert "generate" in result.output
    assert "inspect" in result.output


def _generate(brief_file: Path) -> str:
    """Run `generate` and return the run ID it printed."""
    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])
    match = re.search(r"^Run ID: (\S+)$", result.stdout, re.MULTILINE)
    assert match, result.stdout
    return match.group(1)


def test_generate_prints_the_run_id_and_saves_checkpoints(
    fake: FakeGenerate, brief_file: Path, checkpoint_db: Path
) -> None:
    run_id = _generate(brief_file)

    assert run_id == fake.calls[0]["run_id"]
    assert checkpoint_db.is_file()


def test_inspect_shows_the_state_after_each_node(fake: FakeGenerate, brief_file: Path) -> None:
    run_id = _generate(brief_file)

    result = runner.invoke(cli.app, ["inspect", run_id])

    assert result.exit_code == 0
    assert f"Run {run_id}: 5 checkpoints" in result.stdout
    rows = [line.split() for line in result.stdout.splitlines()]
    header = ["Step", "After", "Variants", "Passed", "Revisions", "Errors"]
    assert rows[2][: len(header)] == header
    after = [row[:2] for row in rows if row and row[0].isdigit()]
    assert after == [
        ["0", "(input)"],
        ["1", "planner"],
        ["2", "writer"],
        ["3", "critic"],
        ["4", "assembler"],
    ]
    # Step 2, after the writer: two variants, none scored yet, due to be critiqued next.
    writer_row = next(row for row in rows if row[:2] == ["2", "writer"])
    assert writer_row[2:6] == ["2", "0/0", "0", "0"]
    assert writer_row[-2:] == ["running", "critic"]
    # The last step has finished: both variants passed, and nothing is due to run.
    assert rows[-1][:2] == ["4", "assembler"]
    assert rows[-1][-2:] == ["complete", "-"]
    assert rows[-1][2:4] == ["2", "2/2"]
    assert "Errors:" not in result.stdout


def test_inspect_lists_the_errors_of_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, brief_file: Path
) -> None:
    monkeypatch.setattr(
        graph, "write_variants", FakeGenerate(GatewayConfigError("ANTHROPIC_API_KEY is not set."))
    )
    run_id = _generate(brief_file)

    result = runner.invoke(cli.app, ["inspect", run_id])

    assert result.exit_code == 0
    assert "Errors:" in result.stdout
    assert "[writer] GatewayConfigError: ANTHROPIC_API_KEY is not set." in result.stdout
    assert "failed" in result.stdout


def test_inspect_step_prints_the_full_state_after_that_step_as_json(
    fake: FakeGenerate, brief_file: Path
) -> None:
    run_id = _generate(brief_file)

    result = runner.invoke(cli.app, ["inspect", run_id, "--step", "2"])

    assert result.exit_code == 0
    state = json.loads(result.stdout)
    assert state["run_id"] == run_id
    assert [v["id"] for v in state["variants"]] == ["search-1", "social-2"]
    assert state["plan"]["angle"] == "save time"
    assert state["brand"]["id"] == "voltride"
    assert state["status"] == "running"


def test_inspect_step_that_does_not_exist_lists_the_steps_there_are(
    fake: FakeGenerate, brief_file: Path
) -> None:
    run_id = _generate(brief_file)

    result = runner.invoke(cli.app, ["inspect", run_id, "--step", "9"])

    assert result.exit_code == 1
    assert "has no step 9 (steps: 0, 1, 2, 3, 4)" in result.output


def test_inspect_an_unknown_run_fails_cleanly(fake: FakeGenerate, brief_file: Path) -> None:
    _generate(brief_file)

    result = runner.invoke(cli.app, ["inspect", "no-such-run"])

    assert result.exit_code == 1
    assert "no checkpoints for run 'no-such-run'" in result.output


def test_inspect_without_a_database_says_so_and_creates_none(checkpoint_db: Path) -> None:
    result = runner.invoke(cli.app, ["inspect", "anything"])

    assert result.exit_code == 1
    assert "no checkpoint database" in result.output
    assert "brandforge generate" in result.output
    assert not checkpoint_db.parent.exists()


def test_generate_fails_cleanly_when_the_checkpoint_path_is_not_usable(
    fake: FakeGenerate, brief_file: Path, checkpoint_db: Path
) -> None:
    checkpoint_db.parent.write_text("a file where the folder should be", encoding="utf-8")

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 1
    assert "generation failed" in result.output
    assert fake.calls == []
