from pathlib import Path

import pytest

from brandforge.prompts.loader import load_prompt, render_prompt


def test_loads_the_shipped_baseline_prompt() -> None:
    assert "{{brand_id}}" in load_prompt("baseline", "v1")


def test_unknown_version_raises() -> None:
    with pytest.raises(FileNotFoundError, match=r"baseline_v999\.md"):
        load_prompt("baseline", "v999")


@pytest.mark.parametrize("version", ["../v1", "v1/../../x", "", "v 1"])
def test_unsafe_version_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="Invalid prompt"):
        load_prompt("baseline", version)


def test_loads_from_a_given_directory(tmp_path: Path) -> None:
    (tmp_path / "demo_v2.md").write_text("Hello {{who}}", encoding="utf-8")
    assert load_prompt("demo", "v2", tmp_path) == "Hello {{who}}"


def test_render_fills_placeholders() -> None:
    assert render_prompt("{{a}} and {{b}}", a="1", b="2") == "1 and 2"


def test_render_does_not_rescan_substituted_values() -> None:
    assert render_prompt("{{a}}", a="{{b}}", b="x") == "{{b}}"


def test_render_missing_value_raises() -> None:
    with pytest.raises(KeyError, match="a"):
        render_prompt("{{a}}")
