"""Build a persistent Chroma collection of approved examples for each brand (BF-30).

One collection per brand, named with the brand id, under ``settings.chroma_dir``. Each example
is one record: the embedded text is the headline, body and call to action, and the channel is
metadata so a later query can filter on it and rebuild an ``Example`` (ADR 0023). Running the
build again replaces the collection, so the index matches the corpus file.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Documents, Embeddable, EmbeddingFunction, Metadata
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from brandforge.config import get_settings
from brandforge.logging import get_logger
from brandforge.models import Example
from brandforge.retrieval.corpus import EXAMPLES_DIR, list_example_brand_ids, load_examples

logger = get_logger(__name__)

# Cosine distance. Chroma normalises the vectors, which is the ranking a retriever wants.
_SPACE = "cosine"

# Stored on every record, and the fields `example_from_metadata` reads back.
METADATA_FIELDS = ("brand_id", "channel", "headline", "body", "cta")


class IndexBuildError(Exception):
    """The index could not be written."""


@dataclass(frozen=True)
class IndexedBrand:
    """One brand whose examples are now in the persistent index."""

    brand_id: str
    collection: str
    count: int
    persist_dir: Path


def example_document(example: Example) -> str:
    """The text that is embedded. The channel is metadata, so it is not in this text."""
    return f"Headline: {example.headline}\nBody: {example.body}\nCTA: {example.cta}"


def example_metadata(example: Example) -> Metadata:
    """The record metadata. Enough to rebuild the `Example` and to filter by channel."""
    return {
        "brand_id": example.brand_id,
        "channel": example.channel,
        "headline": example.headline,
        "body": example.body,
        "cta": example.cta,
    }


def example_from_metadata(metadata: Mapping[str, Any]) -> Example:
    """Rebuild the `Example` that was stored in a record's metadata."""
    return Example.model_validate({key: metadata[key] for key in METADATA_FIELDS})


def example_id(brand_id: str, position: int) -> str:
    """Id of the example at this 1-based position in the brand's corpus file."""
    return f"{brand_id}-{position:04d}"


def default_embedder() -> DefaultEmbeddingFunction:
    """Chroma's local ONNX MiniLM. Constructing it does not download the model."""
    return DefaultEmbeddingFunction()


def open_client(persist_dir: Path) -> ClientAPI:
    """Open the persistent index. Telemetry is off: the index is local and reports nothing."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def index_examples(
    brand_ids: Sequence[str] | None = None,
    *,
    examples_dir: Path = EXAMPLES_DIR,
    persist_dir: Path | None = None,
    embedder: EmbeddingFunction[Documents] | None = None,
) -> list[IndexedBrand]:
    """Embed each brand's approved ads into its own collection.

    `brand_ids` defaults to every brand that has a corpus file. `persist_dir` defaults to
    `settings.chroma_dir`. `embedder` defaults to Chroma's local model; tests pass their own
    so the suite never downloads that model or calls a provider.

    Raises `ExampleLoadError` for an unknown or invalid brand, and `IndexBuildError` when
    the collection cannot be written.
    """
    ids = list(brand_ids) if brand_ids is not None else list_example_brand_ids(examples_dir)
    if not ids:
        return []
    # Load every brand before opening the index, so a bad id writes nothing.
    loaded = [(brand_id, load_examples(brand_id, examples_dir)) for brand_id in ids]

    directory = get_settings().chroma_dir if persist_dir is None else persist_dir
    resolved = embedder if embedder is not None else default_embedder()
    try:
        client = open_client(directory)
    except OSError as exc:
        raise IndexBuildError(f"cannot open the index at {directory}: {exc}") from exc

    return [
        _index_brand(client, brand_id, examples, directory, resolved)
        for brand_id, examples in loaded
    ]


def _collection_names(client: ClientAPI) -> set[str]:
    return {collection.name for collection in client.list_collections()}


def _index_brand(
    client: ClientAPI,
    brand_id: str,
    examples: list[Example],
    persist_dir: Path,
    embedder: EmbeddingFunction[Documents],
) -> IndexedBrand:
    documents = [example_document(example) for example in examples]
    metadatas = [example_metadata(example) for example in examples]
    ids = [example_id(brand_id, position) for position, _ in enumerate(examples, start=1)]

    # Replace the collection. An upsert would keep an ad that has been removed from the file.
    try:
        if brand_id in _collection_names(client):
            client.delete_collection(brand_id)
        collection = client.create_collection(
            name=brand_id,
            # Chroma types this slot for text or images. The index only embeds text.
            embedding_function=cast(EmbeddingFunction[Embeddable], embedder),
            metadata={"hnsw:space": _SPACE, "brand_id": brand_id},
        )
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    except Exception as exc:
        raise IndexBuildError(f"could not index {brand_id}: {exc}") from exc

    count = collection.count()
    if count != len(examples):
        raise IndexBuildError(f"could not index {brand_id}: wrote {count} of {len(examples)}")

    logger.info("indexed_examples", brand_id=brand_id, count=count, path=str(persist_dir))
    return IndexedBrand(
        brand_id=brand_id,
        collection=brand_id,
        count=count,
        persist_dir=persist_dir,
    )
