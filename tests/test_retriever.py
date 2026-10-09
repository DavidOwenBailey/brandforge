"""Retriever node (BF-31): nearest examples into state, and an empty result continues."""

import io
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from brandforge.agents.retriever import retrieve_examples
from brandforge.brands import load_brand
from brandforge.config import Settings
from brandforge.graph import run_graph
from brandforge.logging import configure_logging
from brandforge.models import (
    Brief,
    Channel,
    Critique,
    Example,
    Plan,
    RunState,
    Usage,
    Variant,
    new_run_state,
)
from brandforge.retrieval.query import SearchResult, query_text


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


class FakeSearch:
    """Returns one fixed result and records the arguments the node searched with."""

    def __init__(self, result: SearchResult) -> None:
        self.result = result
        self.brand_ids: list[str] = []
        self.queries: list[str] = []
        self.channels: list[Sequence[Channel]] = []
        self.per_channel: list[int] = []
        self.persist_dirs: list[Path] = []

    def __call__(
        self,
        brand_id: str,
        query: str,
        channels: Sequence[Channel],
        /,
        *,
        per_channel: int,
        persist_dir: Path,
    ) -> SearchResult:
        self.brand_ids.append(brand_id)
        self.queries.append(query)
        self.channels.append(channels)
        self.per_channel.append(per_channel)
        self.persist_dirs.append(persist_dir)
        return self.result


class Boom:
    def __call__(
        self,
        brand_id: str,
        query: str,
        channels: Sequence[Channel],
        /,
        *,
        per_channel: int,
        persist_dir: Path,
    ) -> SearchResult:
        raise RuntimeError("chroma down")


@pytest.fixture(autouse=True)
def index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A chroma path that does not exist, so a forgotten `settings=` opens no real index."""
    path = tmp_path / "chroma"
    monkeypatch.setattr(
        "brandforge.agents.retriever.get_settings",
        lambda: IsolatedSettings(chroma_dir=path),
    )
    return path


@pytest.fixture
def logs() -> Iterator[io.StringIO]:
    """JSON logs for one test, torn down so later tests are not stuck with our handler."""
    stream = io.StringIO()
    root = logging.getLogger()
    package = logging.getLogger("brandforge")
    previous_root = root.level
    previous_package = package.level
    configure_logging(IsolatedSettings(), stream=stream)
    yield stream
    for handler in list(root.handlers):
        if handler.name == "brandforge":
            root.removeHandler(handler)
            handler.close()
    root.setLevel(previous_root)
    package.setLevel(previous_package)


def _events(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _brief() -> Brief:
    return Brief(
        product="Commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search", "email"],
        constraints=["no discounts"],
    )


def _plan() -> Plan:
    return Plan(
        audience="Time-poor commuters",
        angle="Beat the traffic",
        channels=["search", "email"],
        variants_per_channel=1,
    )


def _state(plan: Plan | None = None) -> RunState:
    state = new_run_state(_brief(), load_brand("voltride"))
    state["plan"] = _plan() if plan is None else plan
    return state


def _example(channel: Channel, headline: str) -> Example:
    return Example(brand_id="voltride", channel=channel, headline=headline, body="Body.", cta="Go")


def test_query_text_uses_the_indexed_ad_shape() -> None:
    text = query_text(_brief(), _plan())

    assert text == (
        "Headline: Beat the traffic\n"
        "Body: Commuter e-bike. Time-poor commuters. conversion. no discounts.\n"
        "CTA: Beat the traffic"
    )


def test_query_text_omits_the_constraints_sentence_when_there_are_none() -> None:
    brief = _brief().model_copy(update={"constraints": []})

    text = query_text(brief, _plan())

    assert "no discounts" not in text
    assert text.splitlines()[1] == "Body: Commuter e-bike. Time-poor commuters. conversion."


def test_the_node_searches_by_brand_and_plan_channels_and_returns_the_examples(
    index_dir: Path,
) -> None:
    examples = [_example("search", "Ride"), _example("email", "Your slot")]
    search = FakeSearch(SearchResult(examples=examples, reason="ok"))
    state = _state()

    update = retrieve_examples(state, search=search)

    assert update == {"examples": examples}
    assert search.brand_ids == ["voltride"]
    assert search.queries == [query_text(state["brief"], _plan())]
    assert search.channels == [["search", "email"]]
    assert search.per_channel == [4]
    assert search.persist_dirs == [index_dir]


def test_the_examples_per_channel_setting_is_what_gets_searched(index_dir: Path) -> None:
    search = FakeSearch(SearchResult(examples=[], reason="no_matches"))
    settings = IsolatedSettings(chroma_dir=index_dir, retrieval_examples_per_channel=2)

    retrieve_examples(_state(), settings=settings, search=search)

    assert search.per_channel == [2]
    assert search.persist_dirs == [index_dir]


def test_an_empty_result_is_not_an_error_and_is_logged(logs: io.StringIO) -> None:
    search = FakeSearch(SearchResult(examples=[], reason="no_index"))

    update = retrieve_examples(_state(), search=search)

    assert update == {"examples": []}
    assert "errors" not in update
    warning = next(event for event in _events(logs) if event["event"] == "retrieval_empty")
    assert warning["level"] == "warning"
    assert warning["reason"] == "no_index"
    assert warning["brand_id"] == "voltride"
    assert warning["channels"] == ["search", "email"]


def test_a_hit_is_logged_at_info_and_not_as_empty(logs: io.StringIO) -> None:
    search = FakeSearch(SearchResult(examples=[_example("search", "Ride")], reason="ok"))

    retrieve_examples(_state(), search=search)

    events = _events(logs)
    info = next(event for event in events if event["event"] == "retrieved_examples")
    assert info["level"] == "info"
    assert info["count"] == 1
    assert info["brand_id"] == "voltride"
    assert not any(event["event"] == "retrieval_empty" for event in events)


def test_retrieval_disabled_writes_no_examples_and_does_not_search(logs: io.StringIO) -> None:
    settings = IsolatedSettings(retrieval_enabled=False)

    update = retrieve_examples(_state(), settings=settings, search=Boom())

    assert update == {"examples": []}
    assert "errors" not in update
    events = _events(logs)
    info = next(event for event in events if event["event"] == "retrieval_disabled")
    assert info["level"] == "info"
    assert info["brand_id"] == "voltride"
    assert not any(event["event"] == "retrieval_empty" for event in events)
    assert not any(event["event"] == "retrieved_examples" for event in events)


def test_retrieval_disabled_still_requires_a_plan() -> None:
    state = _state()
    state["plan"] = None
    settings = IsolatedSettings(retrieval_enabled=False)

    with pytest.raises(ValueError, match="needs a plan"):
        retrieve_examples(state, settings=settings, search=Boom())


def test_the_retriever_refuses_to_search_without_a_plan() -> None:
    search = FakeSearch(SearchResult(examples=[], reason="no_index"))
    state = _state()
    state["plan"] = None

    with pytest.raises(ValueError, match="needs a plan"):
        retrieve_examples(state, search=search)

    assert search.brand_ids == []


def test_a_search_failure_propagates() -> None:
    with pytest.raises(RuntimeError, match="chroma down"):
        retrieve_examples(_state(), search=Boom())


def test_a_missing_index_inside_the_graph_continues_with_no_examples(
    logs: io.StringIO, index_dir: Path
) -> None:
    """The real search, against a directory that is not an index. The writer still runs."""
    seen: list[RunState] = []

    def plan(state: RunState) -> dict[str, Any]:
        return {
            "plan": Plan(
                audience="commuters",
                angle="save time",
                channels=list(state["brief"].channels),
                variants_per_channel=1,
            ),
            "usage": Usage(),
        }

    def write(state: RunState) -> dict[str, Any]:
        seen.append(state)
        variant = Variant(id="search-1", channel="search", headline="H", body="B", cta="C")
        return {"variants": [variant], "usage": Usage()}

    def critique(state: RunState) -> dict[str, Any]:
        return {
            "critiques": [
                Critique(variant_id=variant.id, scores={"voice": 5}, overall=5.0, passed=True)
                for variant in state["variants"]
            ],
            "usage": Usage(),
        }

    state = run_graph(
        _brief(),
        load_brand("voltride"),
        plan=plan,
        write=write,
        critique=critique,
        retrieve=retrieve_examples,
        run_id="run-empty",
    )

    assert seen[0]["examples"] == []
    assert state["examples"] == []
    assert state["errors"] == []
    assert state["status"] == "complete"
    assert not index_dir.exists()
    warning = next(event for event in _events(logs) if event["event"] == "retrieval_empty")
    assert warning["reason"] == "no_index"
    assert warning["run_id"] == "run-empty"


def test_retrieval_disabled_inside_the_graph_skips_the_search(
    logs: io.StringIO, index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real node, with the flag off. The search must not run, and the writer still does."""
    monkeypatch.setattr(
        "brandforge.agents.retriever.get_settings",
        lambda: IsolatedSettings(chroma_dir=index_dir, retrieval_enabled=False),
    )
    seen: list[RunState] = []

    def plan(state: RunState) -> dict[str, Any]:
        return {
            "plan": Plan(
                audience="commuters",
                angle="save time",
                channels=list(state["brief"].channels),
                variants_per_channel=1,
            ),
            "usage": Usage(),
        }

    def write(state: RunState) -> dict[str, Any]:
        seen.append(state)
        variant = Variant(id="search-1", channel="search", headline="H", body="B", cta="C")
        return {"variants": [variant], "usage": Usage()}

    def critique(state: RunState) -> dict[str, Any]:
        return {
            "critiques": [
                Critique(variant_id=variant.id, scores={"voice": 5}, overall=5.0, passed=True)
                for variant in state["variants"]
            ],
            "usage": Usage(),
        }

    state = run_graph(
        _brief(),
        load_brand("voltride"),
        plan=plan,
        write=write,
        critique=critique,
        retrieve=retrieve_examples,
        run_id="run-off",
    )

    assert seen[0]["examples"] == []
    assert state["examples"] == []
    assert state["errors"] == []
    assert state["status"] == "complete"
    assert not index_dir.exists()
    events = _events(logs)
    info = next(event for event in events if event["event"] == "retrieval_disabled")
    assert info["level"] == "info"
    assert info["run_id"] == "run-off"
    assert not any(event["event"] == "retrieval_empty" for event in events)
