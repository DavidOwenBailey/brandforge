from pathlib import Path

import pytest

from brandforge.brands import BrandLoadError, list_brand_ids, load_brand

REQUIRED_CRITERIA = {"voice", "clarity", "call_to_action"}


def test_three_shipped_brands_load() -> None:
    ids = list_brand_ids()
    assert len(ids) == 3
    for brand_id in ids:
        brand = load_brand(brand_id)
        assert brand.id == brand_id
        assert {c.name for c in brand.rubric.criteria} >= REQUIRED_CRITERIA


def test_unknown_brand_raises() -> None:
    with pytest.raises(BrandLoadError, match="Unknown brand"):
        load_brand("nope")


def test_path_traversal_is_rejected() -> None:
    with pytest.raises(BrandLoadError):
        load_brand("../brightleaf")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(BrandLoadError, match="invalid YAML"):
        load_brand("bad", brands_dir=tmp_path)


def test_missing_fields_raise(tmp_path: Path) -> None:
    (tmp_path / "thin.yaml").write_text("id: thin\n", encoding="utf-8")
    with pytest.raises(BrandLoadError):
        load_brand("thin", brands_dir=tmp_path)


def test_id_must_match_filename(tmp_path: Path) -> None:
    src = Path(__file__).parents[1] / "src/brandforge/brands/brightleaf.yaml"
    (tmp_path / "other.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(BrandLoadError, match="must match the filename"):
        load_brand("other", brands_dir=tmp_path)


def test_rubric_anchors_must_cover_1_to_5(tmp_path: Path) -> None:
    src = Path(__file__).parents[1] / "src/brandforge/brands/brightleaf.yaml"
    text = src.read_text(encoding="utf-8").replace(
        "        5: Instantly clear", "        6: Instantly clear", 1
    )
    (tmp_path / "brightleaf.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(BrandLoadError):
        load_brand("brightleaf", brands_dir=tmp_path)
