"""Retriever: fetches the nearest approved examples for each channel into state (BF-31).

Embeddings only, no model call. The query is the brief and the plan, embedded in the same
shape as an indexed ad, and each channel in the plan is searched on its own so one channel
cannot crowd out another. The reasoning is in ADR 0024.

An empty result (no index, no collection, or nothing for these channels) is not a failure.
The node logs a warning and returns an empty list, and the writer runs with no examples
section.

`retrieval_enabled` (BF-32, ADR 0025) switches that search off. The node still runs, writes
an empty list and does not open the index, so an eval can compare the two arms.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from brandforge.config import Settings, get_settings
from brandforge.logging import get_logger
from brandforge.models import Channel, RunState
from brandforge.retrieval.query import SearchResult, query_text, search_examples

logger = get_logger(__name__)


class Searcher(Protocol):
    """The slice of `search_examples` this module uses. Tests pass a fake."""

    def __call__(
        self,
        brand_id: str,
        query: str,
        channels: Sequence[Channel],
        /,
        *,
        per_channel: int,
        persist_dir: Path,
    ) -> SearchResult: ...


def retrieve_examples(
    state: RunState,
    *,
    settings: Settings | None = None,
    search: Searcher = search_examples,
) -> dict[str, Any]:
    """Graph node: reads `brief`, `brand` and `plan`; returns `examples`.

    Returns no other keys. An empty result is not an error, so the run stays able to finish
    `complete`. With `retrieval_enabled` off, returns an empty list and does not search.
    Raises `ValueError` when there is no plan, whether or not retrieval is on; handling that
    is the graph's job (BF-22).
    """
    plan = state["plan"]
    if plan is None:
        raise ValueError("The retriever needs a plan; run the planner first.")
    cfg = settings if settings is not None else get_settings()
    if not cfg.retrieval_enabled:
        logger.info("retrieval_disabled", brand_id=state["brand"].id)
        return {"examples": []}
    found = search(
        state["brand"].id,
        query_text(state["brief"], plan),
        plan.channels,
        per_channel=cfg.retrieval_examples_per_channel,
        persist_dir=cfg.chroma_dir,
    )
    if found.reason == "ok":
        logger.info("retrieved_examples", brand_id=state["brand"].id, count=len(found.examples))
    else:
        logger.warning(
            "retrieval_empty",
            brand_id=state["brand"].id,
            channels=list(plan.channels),
            reason=found.reason,
        )
    return {"examples": found.examples}
