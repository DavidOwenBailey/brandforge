"""Search one brand's example collection (BF-31).

The index is built by ``brandforge index`` (BF-30, ADR 0023). A search embeds a query with the
function stored on the collection, then takes the nearest ads for each channel. A missing
index, a missing collection, or a channel with nothing stored is an empty result, not an
error: the writer is specified to run with no examples (ADR 0024).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from pydantic import ValidationError

from brandforge.logging import get_logger
from brandforge.models import Brief, Channel, Example, Plan
from brandforge.retrieval.index import example_from_metadata, open_client

logger = get_logger(__name__)

Reason = Literal["ok", "no_index", "no_collection", "no_matches"]

# Chroma's persistent directory is ready only once this file exists. Opening a path without
# it would create an empty database, and a query must not do that.
_MARKER = "chroma.sqlite3"


@dataclass(frozen=True)
class SearchResult:
    """The examples a search found, nearest first within each channel, and why it may be empty."""

    examples: list[Example]
    reason: Reason


def query_text(brief: Brief, plan: Plan) -> str:
    """The text that is embedded for a search. Same three lines as an indexed ad (ADR 0023).

    The plan carries the angle and the audience, not the product, so the brief supplies the
    product and the constraints. The shape matches ``example_document`` so the local model
    compares an ad with an ad.
    """
    body = f"{brief.product}. {plan.audience}. {brief.objective}."
    if brief.constraints:
        body = f"{body} {'; '.join(brief.constraints)}."
    return f"Headline: {plan.angle}\nBody: {body}\nCTA: {plan.angle}"


def search_examples(
    brand_id: str,
    query: str,
    channels: Sequence[Channel],
    *,
    per_channel: int,
    persist_dir: Path,
) -> SearchResult:
    """The nearest `per_channel` examples for each channel, in `channels` order.

    `persist_dir` is the index directory (`settings.chroma_dir`). Nothing is written: a
    directory with no index is `no_index` and is left untouched. A brand with no collection
    is `no_collection`. A collection that has no usable ad for these channels is `no_matches`.

    Raises when the index is present but cannot be read. A record that cannot be rebuilt into
    an `Example` is skipped and logged, so one bad row does not fail the search.
    """
    if per_channel < 1:
        raise ValueError(f"per_channel must be at least 1, got {per_channel}")
    if not (persist_dir / _MARKER).is_file():
        return SearchResult(examples=[], reason="no_index")

    client = open_client(persist_dir)
    try:
        collection = client.get_collection(brand_id)
    except NotFoundError:
        return SearchResult(examples=[], reason="no_collection")

    examples: list[Example] = []
    for channel in channels:
        examples.extend(
            _matches(
                collection,
                brand_id=brand_id,
                query=query,
                channel=channel,
                per_channel=per_channel,
            )
        )
    if not examples:
        return SearchResult(examples=[], reason="no_matches")
    return SearchResult(examples=examples, reason="ok")


def _matches(
    collection: Collection,
    *,
    brand_id: str,
    query: str,
    channel: Channel,
    per_channel: int,
) -> list[Example]:
    """Nearest examples for one channel. Fewer than `per_channel` when the channel has fewer."""
    count = collection.count()
    if count == 0:
        return []
    # n_results cannot exceed the collection. The channel filter may match still fewer, and
    # then Chroma returns what it has.
    found = collection.query(
        query_texts=[query],
        n_results=min(per_channel, count),
        where={"channel": channel},
    )
    metadatas = found["metadatas"]
    if not metadatas or not metadatas[0]:
        return []

    examples: list[Example] = []
    for metadata in metadatas[0]:
        if metadata is None:
            continue
        example = _rebuild(metadata, brand_id=brand_id, channel=channel)
        if example is not None:
            examples.append(example)
    return examples


def _rebuild(metadata: Mapping[str, object], *, brand_id: str, channel: Channel) -> Example | None:
    """The `Example` stored in one record, or None when the record is not one we can use."""
    try:
        example = example_from_metadata(metadata)
    except (KeyError, ValidationError):
        logger.warning("retrieval_record_unreadable", brand_id=brand_id, channel=channel)
        return None
    if example.brand_id != brand_id or example.channel != channel:
        return None
    return example
