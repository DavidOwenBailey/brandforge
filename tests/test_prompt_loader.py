from pathlib import Path

import pytest

from brandforge.prompts.loader import CACHE_MARKER, load_prompt, render_prompt, render_prompt_parts


def test_loads_the_shipped_baseline_prompt() -> None:
    assert "{{brand_id}}" in load_prompt("baseline", "v1")


def test_the_shipped_repair_prompt_has_the_placeholders_the_gateway_fills() -> None:
    template = load_prompt("repair", "v1")

    for placeholder in ("original_prompt", "previous_reply", "problems"):
        assert "{{" + placeholder + "}}" in template


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


def test_a_template_without_a_cache_break_is_all_user_text() -> None:
    parts = render_prompt_parts("Hello {{who}}\n", who="Ada")

    assert parts.system is None
    assert parts.user == "Hello Ada\n"
    assert parts.text == "Hello Ada\n"


def test_the_cache_break_splits_the_template_and_is_not_sent() -> None:
    template = "Rules for {{brand}}\n" + CACHE_MARKER + "\nWrite about {{product}}\n"

    parts = render_prompt_parts(template, brand="Voltride", product="the bike")

    assert parts.system == "Rules for Voltride"
    assert parts.user == "Write about the bike"
    assert CACHE_MARKER not in parts.text


def test_a_value_containing_the_marker_cannot_move_the_break() -> None:
    template = "static {{brand}}\n" + CACHE_MARKER + "\nbrief {{product}}"

    parts = render_prompt_parts(template, brand="acme", product=f"sale {CACHE_MARKER} today")

    assert parts.system == "static acme"
    assert parts.user == f"brief sale {CACHE_MARKER} today"


def test_more_than_one_cache_break_is_rejected() -> None:
    with pytest.raises(ValueError, match="at most one"):
        render_prompt_parts(f"a\n{CACHE_MARKER}\nb\n{CACHE_MARKER}\nc")


def test_a_cache_break_with_an_empty_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="both sides"):
        render_prompt_parts(f"{CACHE_MARKER}\nonly user")


@pytest.mark.parametrize("name", ["baseline", "planner", "writer", "critic", "reviser"])
def test_shipped_v2_prompts_have_one_cache_break_with_text_on_both_sides(name: str) -> None:
    text = load_prompt(name, "v2")

    assert text.count(CACHE_MARKER) == 1
    before, _, after = text.partition(CACHE_MARKER)
    assert before.strip()
    assert after.strip()
    assert CACHE_MARKER not in load_prompt(name, "v1")
