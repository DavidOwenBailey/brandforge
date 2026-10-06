"""The shipped sample briefs must all load, and together cover every channel and objective."""

from pathlib import Path

import pytest

import brandforge
from brandforge.brands import list_brand_ids
from brandforge.interfaces.cli import load_brief

BRIEFS_DIR = Path(brandforge.__file__).parent / "briefs"
BRIEF_FILES = sorted(BRIEFS_DIR.glob("*.yaml"))


def test_ten_sample_briefs_ship() -> None:
    assert len(BRIEF_FILES) == 10


@pytest.mark.parametrize("path", BRIEF_FILES, ids=lambda p: p.stem)
def test_brief_is_valid_and_named_for_a_known_brand(path: Path) -> None:
    load_brief(path)
    assert path.stem.split("_")[0] in list_brand_ids()


def test_every_brand_has_at_least_three_briefs() -> None:
    prefixes = [p.stem.split("_")[0] for p in BRIEF_FILES]
    for brand_id in list_brand_ids():
        assert prefixes.count(brand_id) >= 3


def test_briefs_cover_all_channels_and_objectives() -> None:
    briefs = [load_brief(p) for p in BRIEF_FILES]
    assert {c for b in briefs for c in b.channels} == {"search", "social", "display", "email"}
    assert {b.objective for b in briefs} == {"awareness", "consideration", "conversion"}
