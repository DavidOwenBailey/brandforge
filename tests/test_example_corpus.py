"""The approved example corpus: 15 fictional ads per brand, valid against Example."""

from collections import Counter
from pathlib import Path

import pytest

from brandforge.brands import list_brand_ids, load_brand
from brandforge.models import Example
from brandforge.retrieval import ExampleLoadError, list_example_brand_ids, load_examples

# Tightest limits already used by the sample briefs, so an approved ad is safe to imitate.
# Social has no brief limit.
HEADLINE_LIMITS = {"search": 30, "display": 40, "email": 50}
ADS_PER_BRAND = 15
MIN_PER_CHANNEL = 3


def test_corpus_covers_every_shipped_brand() -> None:
    assert list_example_brand_ids() == list_brand_ids()


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_fifteen_approved_ads_load_as_examples(brand_id: str) -> None:
    examples = load_examples(brand_id)
    assert len(examples) == ADS_PER_BRAND
    assert all(isinstance(item, Example) for item in examples)
    assert all(item.brand_id == brand_id for item in examples)


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_each_brand_covers_every_channel(brand_id: str) -> None:
    counts = Counter(item.channel for item in load_examples(brand_id))
    assert set(counts) == {"search", "social", "display", "email"}
    assert all(count >= MIN_PER_CHANNEL for count in counts.values())


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_headlines_and_bodies_are_unique(brand_id: str) -> None:
    examples = load_examples(brand_id)
    headlines = [item.headline for item in examples]
    bodies = [item.body for item in examples]
    assert len(set(headlines)) == len(headlines)
    assert len(set(bodies)) == len(bodies)


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_headlines_fit_the_channel_limits_used_in_briefs(brand_id: str) -> None:
    for example in load_examples(brand_id):
        limit = HEADLINE_LIMITS.get(example.channel)
        if limit is None:
            continue
        length = len(example.headline)
        assert length <= limit, (
            f"{brand_id} {example.channel} headline {example.headline!r} "
            f"is {length} characters; limit is {limit}"
        )


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_banned_words_are_absent(brand_id: str) -> None:
    brand = load_brand(brand_id)
    for example in load_examples(brand_id):
        text = f"{example.headline}\n{example.body}\n{example.cta}".casefold()
        for banned in brand.banned_words:
            assert banned.casefold() not in text, f"{brand_id} uses banned word {banned!r}"


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_exclamation_marks_follow_the_brand_rules(brand_id: str) -> None:
    """Ledgerly forbids exclamation marks. Voltride allows at most one per ad."""
    for example in load_examples(brand_id):
        count = f"{example.headline}{example.body}{example.cta}".count("!")
        if brand_id == "voltride":
            assert count <= 1
        elif brand_id == "ledgerly":
            assert count == 0


def test_unknown_brand_raises() -> None:
    with pytest.raises(ExampleLoadError, match="Unknown brand"):
        load_examples("nope")


def test_path_traversal_is_rejected() -> None:
    with pytest.raises(ExampleLoadError):
        load_examples("../brightleaf")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("brand_id: [unclosed", encoding="utf-8")
    with pytest.raises(ExampleLoadError, match="invalid YAML"):
        load_examples("bad", examples_dir=tmp_path)


def test_brand_id_must_match_filename(tmp_path: Path) -> None:
    src = Path(__file__).parents[1] / "src/brandforge/retrieval/examples/brightleaf.yaml"
    (tmp_path / "other.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ExampleLoadError, match="must match the filename"):
        load_examples("other", examples_dir=tmp_path)


def test_empty_examples_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "thin.yaml").write_text("brand_id: thin\nexamples: []\n", encoding="utf-8")
    with pytest.raises(ExampleLoadError):
        load_examples("thin", examples_dir=tmp_path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    text = """
brand_id: tiny
examples:
  - channel: search
    headline: Hello
    body: A sentence.
    cta: Go
    note: extra
"""
    (tmp_path / "tiny.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ExampleLoadError):
        load_examples("tiny", examples_dir=tmp_path)


def test_bad_channel_is_rejected(tmp_path: Path) -> None:
    text = """
brand_id: tiny
examples:
  - channel: billboard
    headline: Hello
    body: A sentence.
    cta: Go
"""
    (tmp_path / "tiny.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ExampleLoadError):
        load_examples("tiny", examples_dir=tmp_path)
