"""The example index: one persistent Chroma collection per brand, no model download."""

import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    register_embedding_function,
)
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from pydantic_settings import SettingsConfigDict
from typer.testing import CliRunner

from brandforge.agents.retriever import retrieve_examples
from brandforge.brands import load_brand
from brandforge.config import Settings
from brandforge.interfaces import cli
from brandforge.models import Brief, Channel, Example, Plan, new_run_state
from brandforge.retrieval import ExampleLoadError, list_example_brand_ids, load_examples
from brandforge.retrieval import index as index_module
from brandforge.retrieval.index import (
    IndexBuildError,
    example_document,
    example_from_metadata,
    example_id,
    example_metadata,
    index_examples,
    open_client,
)
from brandforge.retrieval.query import SearchResult, search_examples

runner = CliRunner()

ADS_PER_BRAND = 15


def _vector(text: str) -> list[float]:
    """A tiny deterministic vector. Identical text matches; different text does not have to."""
    vec = [0.0] * 16
    for token in text.casefold().split():
        digest = hashlib.sha256(token.encode()).digest()
        for offset, byte in enumerate(digest):
            vec[offset % 16] += byte / 255
    return vec


@register_embedding_function
class HashEmbedder(EmbeddingFunction[Documents]):
    """Stands in for the ONNX model. Registered so a reopened collection can resolve it."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [np.asarray(_vector(text), dtype=np.float32) for text in input]

    @staticmethod
    def name() -> str:
        return "brandforge-test-hash"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashEmbedder":
        return HashEmbedder()

    def default_space(self) -> Literal["cosine"]:
        return "cosine"


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


@pytest.fixture(autouse=True)
def no_onnx_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real embedder downloads a model on first use. No unit test may reach it."""

    def refuse(self: ONNXMiniLM_L6_V2, input: Documents) -> Embeddings:
        raise AssertionError("unit tests must not run the ONNX embedder")

    monkeypatch.setattr(ONNXMiniLM_L6_V2, "__call__", refuse)


def _write_brand(directory: Path, brand_id: str, ads: list[tuple[str, str, str, str]]) -> None:
    lines = [f"brand_id: {brand_id}", "examples:"]
    for channel, headline, body, cta in ads:
        lines += [
            f"  - channel: {channel}",
            f"    headline: {headline}",
            f"    body: {body}",
            f"    cta: {cta}",
        ]
    (directory / f"{brand_id}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_default_embedder_is_chromas_local_model() -> None:
    embedder = index_module.default_embedder()
    assert isinstance(embedder, DefaultEmbeddingFunction)
    assert embedder.name() == "default"
    assert ONNXMiniLM_L6_V2.MODEL_NAME == "all-MiniLM-L6-v2"


def test_channel_is_metadata_and_the_copy_is_the_document() -> None:
    example = Example(
        brand_id="voltride",
        channel="search",
        headline="Ride the commute",
        body="Sit upright.",
        cta="See the commuter",
    )
    document = example_document(example)
    assert "Channel" not in document
    assert "Ride the commute" in document
    assert example_metadata(example)["channel"] == "search"
    assert example_from_metadata(example_metadata(example)) == example


def test_shipped_corpus_indexes_one_persistent_collection_per_brand(tmp_path: Path) -> None:
    store = tmp_path / "chroma"
    indexed = index_examples(persist_dir=store, embedder=HashEmbedder())

    assert [item.brand_id for item in indexed] == list_example_brand_ids()
    assert [item.collection for item in indexed] == list_example_brand_ids()
    assert all(item.count == ADS_PER_BRAND for item in indexed)
    assert (store / "chroma.sqlite3").is_file()

    client = open_client(store)
    assert (
        sorted(collection.name for collection in client.list_collections())
        == list_example_brand_ids()
    )
    for brand_id in list_example_brand_ids():
        stored = client.get_collection(brand_id).get(include=["documents", "metadatas"])
        documents = stored["documents"]
        metadatas = stored["metadatas"]
        assert documents is not None
        assert metadatas is not None
        examples = load_examples(brand_id)
        rows = sorted(
            zip(stored["ids"], documents, metadatas, strict=True),
            key=lambda row: row[0],
        )
        assert [row[0] for row in rows] == [
            example_id(brand_id, i) for i in range(1, ADS_PER_BRAND + 1)
        ]
        assert [row[1] for row in rows] == [example_document(example) for example in examples]
        assert [example_from_metadata(row[2]) for row in rows] == examples


def test_reindex_drops_examples_removed_from_the_corpus(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    store = tmp_path / "chroma"
    ads = [
        ("search", "One", "First body.", "Go"),
        ("email", "Two", "Second body.", "Stop"),
    ]
    _write_brand(examples, "acme", ads)
    first = index_examples(
        ["acme"], examples_dir=examples, persist_dir=store, embedder=HashEmbedder()
    )
    assert first[0].count == 2

    _write_brand(examples, "acme", ads[:1])
    second = index_examples(
        ["acme"], examples_dir=examples, persist_dir=store, embedder=HashEmbedder()
    )

    assert second[0].count == 1
    stored = open_client(store).get_collection("acme").get()
    assert stored["ids"] == ["acme-0001"]


def test_a_reopened_collection_resolves_the_embedder_and_filters_by_channel(
    tmp_path: Path,
) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    store = tmp_path / "chroma"
    _write_brand(
        examples,
        "acme",
        [
            ("search", "Ride the commute", "Sit upright and go.", "See it"),
            ("email", "Your slot is Saturday", "Twenty minutes in the store.", "Book it"),
        ],
    )
    index_examples(["acme"], examples_dir=examples, persist_dir=store, embedder=HashEmbedder())

    # A new client, and no embedder passed in: the collection must use the one it stored.
    collection = open_client(store).get_collection("acme")
    document = example_document(
        Example(
            brand_id="acme",
            channel="search",
            headline="Ride the commute",
            body="Sit upright and go.",
            cta="See it",
        )
    )
    nearest = collection.query(query_texts=[document], n_results=1)
    assert nearest["ids"] == [["acme-0001"]]

    filtered = collection.query(query_texts=[document], n_results=1, where={"channel": "email"})
    assert filtered["ids"] == [["acme-0002"]]


def test_unknown_brand_raises_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ExampleLoadError, match="Unknown brand"):
        index_examples(["nope"], persist_dir=tmp_path / "chroma", embedder=HashEmbedder())
    assert not (tmp_path / "chroma").exists()


def test_an_embedder_failure_becomes_an_index_build_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(self: HashEmbedder, input: Documents) -> Embeddings:
        raise RuntimeError("embed failed")

    monkeypatch.setattr(HashEmbedder, "__call__", refuse)

    with pytest.raises(IndexBuildError, match="embed failed"):
        index_examples(["voltride"], persist_dir=tmp_path / "chroma", embedder=HashEmbedder())


def test_an_injected_embedder_is_used_instead_of_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unavailable() -> HashEmbedder:
        raise AssertionError("the default embedder should not be built")

    monkeypatch.setattr(index_module, "default_embedder", unavailable)
    indexed = index_examples(["voltride"], persist_dir=tmp_path / "chroma", embedder=HashEmbedder())
    assert indexed[0].count == ADS_PER_BRAND


def test_the_default_embedder_is_built_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[HashEmbedder] = []

    def factory() -> HashEmbedder:
        embedder = HashEmbedder()
        built.append(embedder)
        return embedder

    monkeypatch.setattr(index_module, "default_embedder", factory)
    index_examples(["voltride"], persist_dir=tmp_path / "chroma")
    assert len(built) == 1


def test_the_directory_defaults_to_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "from-settings"
    monkeypatch.setattr(index_module, "get_settings", lambda: IsolatedSettings(chroma_dir=store))
    monkeypatch.setattr(index_module, "default_embedder", HashEmbedder)
    indexed = index_examples(["voltride"])
    assert indexed[0].persist_dir == store
    assert (store / "chroma.sqlite3").is_file()


def test_an_empty_corpus_directory_indexes_nothing(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    assert index_examples(examples_dir=examples, persist_dir=tmp_path / "chroma") == []
    assert not (tmp_path / "chroma").exists()


def test_cli_indexes_every_brand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = tmp_path / "chroma"
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings(chroma_dir=store))
    monkeypatch.setattr(index_module, "default_embedder", HashEmbedder)

    result = runner.invoke(cli.app, ["index"])

    assert result.exit_code == 0
    for brand_id in list_example_brand_ids():
        assert (
            f"Indexed {ADS_PER_BRAND} examples for {brand_id} into collection '{brand_id}'"
            in result.stdout
        )
        assert str(store) in result.stdout
    assert (
        sorted(collection.name for collection in open_client(store).list_collections())
        == list_example_brand_ids()
    )


def test_cli_indexes_one_brand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = tmp_path / "chroma"
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings(chroma_dir=store))
    monkeypatch.setattr(index_module, "default_embedder", HashEmbedder)

    result = runner.invoke(cli.app, ["index", "--brand", "voltride"])

    assert result.exit_code == 0
    assert result.stdout.count("Indexed") == 1
    assert "voltride" in result.stdout
    assert [collection.name for collection in open_client(store).list_collections()] == ["voltride"]


def test_cli_unknown_brand_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = tmp_path / "chroma"
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings(chroma_dir=store))
    monkeypatch.setattr(index_module, "default_embedder", HashEmbedder)

    result = runner.invoke(cli.app, ["index", "--brand", "nope"])

    assert result.exit_code == 1
    assert "Unknown brand" in result.output
    assert not store.exists()


# --- search (BF-31) -----------------------------------------------------------------


def _index_ads(tmp_path: Path, brand_id: str, ads: list[tuple[str, str, str, str]]) -> Path:
    examples = tmp_path / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "chroma"
    _write_brand(examples, brand_id, ads)
    index_examples([brand_id], examples_dir=examples, persist_dir=store, embedder=HashEmbedder())
    return store


def _ad(brand_id: str, channel: Channel, headline: str, body: str, cta: str) -> Example:
    return Example(brand_id=brand_id, channel=channel, headline=headline, body=body, cta=cta)


def test_search_ranks_the_matching_ad_first_and_stays_on_the_channel(tmp_path: Path) -> None:
    ride = _ad("acme", "search", "Ride the commute", "Sit upright and go.", "See it")
    fold = _ad("acme", "search", "Fold it under the desk", "The fold is for the train.", "Compare")
    email = _ad("acme", "email", "Your slot is Saturday", "Twenty minutes in the store.", "Book it")
    store = _index_ads(
        tmp_path,
        "acme",
        [
            (ride.channel, ride.headline, ride.body, ride.cta),
            (fold.channel, fold.headline, fold.body, fold.cta),
            (email.channel, email.headline, email.body, email.cta),
        ],
    )

    found = search_examples(
        "acme", example_document(fold), ["search"], per_channel=2, persist_dir=store
    )

    assert found.reason == "ok"
    assert found.examples[0] == fold
    assert found.examples == [fold, ride]


def test_search_keeps_the_channels_in_the_order_it_was_asked(tmp_path: Path) -> None:
    ride = _ad("acme", "search", "Ride the commute", "Sit upright and go.", "See it")
    email = _ad("acme", "email", "Your slot is Saturday", "Twenty minutes in the store.", "Book it")
    store = _index_ads(
        tmp_path,
        "acme",
        [
            (ride.channel, ride.headline, ride.body, ride.cta),
            (email.channel, email.headline, email.body, email.cta),
        ],
    )

    found = search_examples(
        "acme",
        example_document(email),
        ["email", "search"],
        per_channel=4,
        persist_dir=store,
    )

    assert [item.channel for item in found.examples] == ["email", "search"]
    assert found.examples[0] == email


def test_search_does_not_borrow_examples_from_another_channel(tmp_path: Path) -> None:
    email = _ad("acme", "email", "Your slot is Saturday", "Twenty minutes in the store.", "Book it")
    store = _index_ads(tmp_path, "acme", [(email.channel, email.headline, email.body, email.cta)])

    found = search_examples(
        "acme", example_document(email), ["search"], per_channel=4, persist_dir=store
    )

    assert found == SearchResult(examples=[], reason="no_matches")


def test_search_caps_each_channel_and_returns_fewer_when_the_channel_has_fewer(
    tmp_path: Path,
) -> None:
    ads = [
        _ad("acme", "search", f"Headline {n}", f"Body {n} is longer than the headline.", "Go")
        for n in range(6)
    ]
    store = _index_ads(tmp_path, "acme", [(ad.channel, ad.headline, ad.body, ad.cta) for ad in ads])

    capped = search_examples(
        "acme", example_document(ads[0]), ["search"], per_channel=4, persist_dir=store
    )
    one = _index_ads(
        tmp_path / "one",
        "acme",
        [(ads[0].channel, ads[0].headline, ads[0].body, ads[0].cta)],
    )
    short = search_examples("acme", "Headline", ["search"], per_channel=4, persist_dir=one)

    assert capped.reason == "ok"
    assert len(capped.examples) == 4
    assert capped.examples[0] == ads[0]
    assert short.examples == [ads[0]]


def test_a_missing_index_is_empty_and_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()

    assert search_examples(
        "acme", "query", ["search"], per_channel=4, persist_dir=missing
    ) == SearchResult(examples=[], reason="no_index")
    assert search_examples(
        "acme", "query", ["search"], per_channel=4, persist_dir=empty
    ) == SearchResult(examples=[], reason="no_index")
    assert not missing.exists()
    assert list(empty.iterdir()) == []


def test_a_brand_with_no_collection_is_empty(tmp_path: Path) -> None:
    store = _index_ads(tmp_path, "acme", [("search", "One", "A body.", "Go")])

    found = search_examples("other", "query", ["search"], per_channel=4, persist_dir=store)

    assert found == SearchResult(examples=[], reason="no_collection")


def test_search_does_not_return_another_brands_examples(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    store = tmp_path / "chroma"
    _write_brand(examples, "acme", [("search", "Acme rides", "Acme body.", "Go")])
    _write_brand(examples, "other", [("search", "Other rides", "Other body.", "Stop")])
    index_examples(
        ["acme", "other"], examples_dir=examples, persist_dir=store, embedder=HashEmbedder()
    )

    found = search_examples("acme", "rides", ["search"], per_channel=4, persist_dir=store)

    assert [item.brand_id for item in found.examples] == ["acme"]
    assert found.examples[0].headline == "Acme rides"


def test_a_record_that_cannot_be_rebuilt_is_skipped(tmp_path: Path) -> None:
    good = _ad("acme", "search", "Ride the commute", "Sit upright and go.", "See it")
    store = _index_ads(tmp_path, "acme", [(good.channel, good.headline, good.body, good.cta)])
    open_client(store).get_collection("acme").add(
        ids=["bad"],
        documents=["Headline: broken\nBody: broken.\nCTA: broken"],
        metadatas=[{"channel": "search"}],
    )

    found = search_examples(
        "acme", example_document(good), ["search"], per_channel=4, persist_dir=store
    )

    assert found.examples == [good]


def test_per_channel_must_be_at_least_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="per_channel"):
        search_examples("acme", "query", ["search"], per_channel=0, persist_dir=tmp_path)


def test_the_retriever_node_writes_one_brands_examples_for_each_plan_channel(
    tmp_path: Path,
) -> None:
    store = tmp_path / "chroma"
    index_examples(["voltride"], persist_dir=store, embedder=HashEmbedder())
    state = new_run_state(
        Brief(
            product="Commuter e-bike",
            audience="city commuters",
            objective="conversion",
            channels=["search", "email"],
        ),
        load_brand("voltride"),
    )
    state["plan"] = Plan(
        audience="Time-poor commuters",
        angle="Beat the traffic",
        channels=["search", "email"],
        variants_per_channel=1,
    )

    update = retrieve_examples(
        state, settings=IsolatedSettings(chroma_dir=store, retrieval_examples_per_channel=4)
    )

    assert set(update) == {"examples"}
    examples = update["examples"]
    assert examples
    assert all(item.brand_id == "voltride" for item in examples)
    seen: list[str] = []
    for item in examples:
        if not seen or seen[-1] != item.channel:
            seen.append(item.channel)
    assert seen == ["search", "email"]
    assert 1 <= sum(item.channel == "search" for item in examples) <= 4
    assert 1 <= sum(item.channel == "email" for item in examples) <= 4
