"""Load the approved example-ad corpus for one brand (BF-29).

Each brand has one YAML file in ``examples/``, named ``<brand_id>.yaml``. The file
holds ``brand_id`` and a list of ads. Each ad has ``channel``, ``headline``, ``body``
and ``cta``. The loader validates the file and returns ``Example`` models, which is
the shape the retriever will later put in run state. ``brand_id`` is written once, on
the file, and stamped onto every example.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from brandforge.models import Channel, Example, NonEmptyStr

EXAMPLES_DIR = Path(__file__).parent / "examples"


class ExampleLoadError(Exception):
    """An example file is missing, malformed or fails validation."""


class _CorpusExample(BaseModel):
    """One ad in the corpus file. ``brand_id`` lives on the file, not on the ad."""

    model_config = ConfigDict(extra="forbid")

    channel: Channel
    headline: NonEmptyStr
    body: NonEmptyStr
    cta: NonEmptyStr


class _CorpusFile(BaseModel):
    """The on-disk shape of one brand's approved ads."""

    model_config = ConfigDict(extra="forbid")

    brand_id: NonEmptyStr
    examples: list[_CorpusExample] = Field(min_length=1)


def list_example_brand_ids(examples_dir: Path = EXAMPLES_DIR) -> list[str]:
    return sorted(p.stem for p in examples_dir.glob("*.yaml"))


def load_examples(brand_id: str, examples_dir: Path = EXAMPLES_DIR) -> list[Example]:
    """Load and validate one brand's approved ads, in file order.

    Raises `ExampleLoadError` for an unknown brand, invalid YAML, a schema violation
    or a brand id that does not match the filename. Checking the id against the
    directory listing also blocks path tricks such as ``../x``.
    """
    if brand_id not in list_example_brand_ids(examples_dir):
        known = ", ".join(list_example_brand_ids(examples_dir))
        raise ExampleLoadError(f"Unknown brand '{brand_id}'. Known brands: {known}")

    path = examples_dir / f"{brand_id}.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExampleLoadError(f"{path.name}: invalid YAML: {exc}") from exc

    try:
        corpus = _CorpusFile.model_validate(raw)
    except ValidationError as exc:
        raise ExampleLoadError(f"{path.name}: {exc}") from exc

    if corpus.brand_id != path.stem:
        raise ExampleLoadError(f"{path.name}: brand_id '{corpus.brand_id}' must match the filename")

    return [
        Example(
            brand_id=corpus.brand_id,
            channel=item.channel,
            headline=item.headline,
            body=item.body,
            cta=item.cta,
        )
        for item in corpus.examples
    ]
