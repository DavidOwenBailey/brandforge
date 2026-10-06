from pathlib import Path

import yaml
from pydantic import ValidationError

from brandforge.models import BrandProfile

BRANDS_DIR = Path(__file__).parent


class BrandLoadError(Exception):
    """A brand file is missing, malformed or fails validation."""


def list_brand_ids(brands_dir: Path = BRANDS_DIR) -> list[str]:
    return sorted(p.stem for p in brands_dir.glob("*.yaml"))


def load_brand(brand_id: str, brands_dir: Path = BRANDS_DIR) -> BrandProfile:
    # Checking against the directory listing also blocks path tricks like "../x"
    if brand_id not in list_brand_ids(brands_dir):
        known = ", ".join(list_brand_ids(brands_dir))
        raise BrandLoadError(f"Unknown brand '{brand_id}'. Known brands: {known}")

    path = brands_dir / f"{brand_id}.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BrandLoadError(f"{path.name}: invalid YAML: {exc}") from exc

    try:
        brand = BrandProfile.model_validate(raw)
    except ValidationError as exc:
        raise BrandLoadError(f"{path.name}: {exc}") from exc

    if brand.id != path.stem:
        raise BrandLoadError(f"{path.name}: id '{brand.id}' must match the filename")
    return brand
