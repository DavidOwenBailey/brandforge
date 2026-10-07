"""CLI tests. The model is faked by replacing the planner and writer in the graph module."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from brandforge import __version__, graph
from brandforge.interfaces import cli
from brandforge.llm.base import GatewayConfigError
from brandforge.models import Plan, RunState, Usage, Variant

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


@pytest.fixture(autouse=True)
def fake_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "plan_brief", _fake_plan)


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


def test_gateway_error_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, brief_file: Path
) -> None:
    monkeypatch.setattr(
        graph,
        "write_variants",
        FakeGenerate(GatewayConfigError("ANTHROPIC_API_KEY is not set.")),
    )

    result = runner.invoke(cli.app, ["generate", "--brand", "voltride", "--brief", str(brief_file)])

    assert result.exit_code == 1
    assert "generation failed: ANTHROPIC_API_KEY is not set." in result.output


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(cli.app, [])

    assert "generate" in result.output
