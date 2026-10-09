"""Pydantic data contracts for every boundary in BrandForge."""

import time
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[int, Field(ge=1, le=5)]

Channel = Literal["search", "social", "display", "email"]
Objective = Literal["awareness", "consideration", "conversion"]


class _Contract(BaseModel):
    """Base for all contracts: unknown fields are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


class Brief(_Contract):
    product: NonEmptyStr
    audience: NonEmptyStr
    objective: Objective
    channels: list[Channel] = Field(min_length=1)
    constraints: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("channels")
    @classmethod
    def _channels_unique(cls, v: list[Channel]) -> list[Channel]:
        if len(set(v)) != len(v):
            raise ValueError("channels must not contain duplicates")
        return v


class RubricCriterion(_Contract):
    name: NonEmptyStr
    description: NonEmptyStr
    anchors: dict[Score, NonEmptyStr]  # score level -> concrete description

    @model_validator(mode="after")
    def _anchors_cover_all_levels(self) -> "RubricCriterion":
        if set(self.anchors) != {1, 2, 3, 4, 5}:
            raise ValueError("anchors must define exactly the levels 1 to 5")
        return self


class Rubric(_Contract):
    version: NonEmptyStr  # recorded in results alongside the brand version
    criteria: list[RubricCriterion] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _names_unique(cls, v: list[RubricCriterion]) -> list[RubricCriterion]:
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError("criterion names must be unique")
        return v


class BrandProfile(_Contract):
    id: NonEmptyStr
    version: NonEmptyStr  # recorded in results so eval runs are reproducible
    voice: list[NonEmptyStr] = Field(min_length=1)
    do: list[NonEmptyStr] = Field(default_factory=list)
    dont: list[NonEmptyStr] = Field(default_factory=list)
    banned_words: list[NonEmptyStr] = Field(default_factory=list)
    rubric: Rubric


class Variant(_Contract):
    id: NonEmptyStr
    channel: Channel
    headline: NonEmptyStr
    body: str
    cta: NonEmptyStr


class Critique(_Contract):
    variant_id: NonEmptyStr
    scores: dict[NonEmptyStr, Score] = Field(min_length=1)  # criterion -> 1..5
    overall: float = Field(ge=1.0, le=5.0)
    passed: bool
    fixes: list[NonEmptyStr] = Field(default_factory=list)


class NodeUsage(_Contract):
    """What one graph node spent. Repeat visits of the same node are summed (BF-27)."""

    node: NonEmptyStr
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_write_tokens
            + self.cache_read_tokens
        )


def _merge_nodes(left: list[NodeUsage], right: list[NodeUsage]) -> list[NodeUsage]:
    """Sum entries that name the same node, keeping the order nodes were first seen."""
    merged: dict[str, NodeUsage] = {}
    order: list[str] = []
    for item in (*left, *right):
        current = merged.get(item.node)
        if current is None:
            order.append(item.node)
            merged[item.node] = item
            continue
        merged[item.node] = NodeUsage(
            node=item.node,
            input_tokens=current.input_tokens + item.input_tokens,
            output_tokens=current.output_tokens + item.output_tokens,
            cache_write_tokens=current.cache_write_tokens + item.cache_write_tokens,
            cache_read_tokens=current.cache_read_tokens + item.cache_read_tokens,
            cost_usd=current.cost_usd + item.cost_usd,
        )
    return [merged[name] for name in order]


class Usage(_Contract):
    """Tokens and USD for one call, one node, or a whole run.

    `input_tokens` are the uncached input tokens. Cache writes and cache reads are counted
    separately because providers bill them at different rates (BF-27). `total_tokens` includes
    all four, so the run's token budget counts cached context too. `nodes` is empty until the
    graph attributes a node's spend; the reducer sums a node that runs more than once.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    nodes: list[NodeUsage] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_write_tokens
            + self.cache_read_tokens
        )

    def __add__(self, other: "Usage") -> "Usage":
        # The reducer that accumulates usage across nodes (see `RunState`).
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            nodes=_merge_nodes(self.nodes, other.nodes),
        )

    def attributed_to(self, node: str) -> "Usage":
        """A copy that records these totals under `node` for the cost report.

        The graph calls this once, on the usage a node returns. A usage that is already
        attributed, or that spent nothing, is returned unchanged so it is not listed twice
        and a zero-cost node does not add an empty row.
        """
        if self.nodes or (self.total_tokens == 0 and self.cost_usd == 0.0):
            return self
        return self.model_copy(
            update={
                "nodes": [
                    NodeUsage(
                        node=node,
                        input_tokens=self.input_tokens,
                        output_tokens=self.output_tokens,
                        cache_write_tokens=self.cache_write_tokens,
                        cache_read_tokens=self.cache_read_tokens,
                        cost_usd=self.cost_usd,
                    )
                ]
            }
        )


class Plan(_Contract):
    """The planner's structured reading of a brief (written by the planner, BF-13)."""

    audience: NonEmptyStr
    angle: NonEmptyStr
    channels: list[Channel] = Field(min_length=1)
    variants_per_channel: int = Field(ge=1, le=10)


class Example(_Contract):
    """An approved piece of copy used as a style reference.

    The approved corpus is YAML under ``retrieval/examples`` (BF-29). ``brandforge index``
    embeds that corpus into one persistent Chroma collection per brand (BF-30). The retriever
    (BF-31) selects from the index and puts the matches in state.
    """

    brand_id: NonEmptyStr
    channel: Channel
    headline: NonEmptyStr
    body: str
    cta: NonEmptyStr


class RunError(_Contract):
    """A failure recorded in state so the run can still end in a defined status.

    `fatal` is set by the graph's error edges (BF-22) when a node raised and the run was sent to
    the assembler with what exists. A non-fatal error is a warning the run carried on past, such
    as the writer getting fewer variants than asked for.
    """

    node: NonEmptyStr
    message: NonEmptyStr
    fatal: bool = False


RunStatus = Literal["running", "complete", "partial", "failed"]
FinalStatus = Literal["complete", "partial", "failed"]  # what a finished run can end as


class VariantResult(_Contract):
    """One variant in the final result, with its latest critique and whether it is flagged."""

    variant: Variant
    critique: Critique | None  # None: the variant was never scored
    flagged: bool  # True: it has no passing critique, so it is not shown to be on brand


class RunResult(_Contract):
    """What a finished run hands back, packaged by the assembler (BF-18).

    It carries the brand and rubric versions so any result can be tied to the exact profile
    that produced it, and the Langfuse trace ID (BF-25) so it can be tied to how it was produced.
    `trace_id` is `None` when the run was not traced.
    """

    run_id: NonEmptyStr
    status: FinalStatus
    brand_id: NonEmptyStr
    brand_version: NonEmptyStr
    rubric_version: NonEmptyStr
    variants: list[VariantResult]
    revision_count: int = Field(ge=0)
    errors: list[RunError]
    usage: Usage
    trace_id: str | None = None

    @property
    def flagged_count(self) -> int:
        return sum(1 for item in self.variants if item.flagged)


def add_usage(left: Usage, right: Usage) -> Usage:
    """LangGraph reducer: nodes return the usage of their own calls, state keeps the sum."""
    return left + right


def add_errors(left: list[RunError], right: list[RunError]) -> list[RunError]:
    """LangGraph reducer: nodes append errors rather than overwrite earlier ones."""
    return [*left, *right]


class RunState(TypedDict):
    """The single source of truth for one run.

    Agents return only the keys they own. `usage` and `errors` carry reducers, so a node
    returns just its own usage or its new errors and the graph accumulates them.
    """

    run_id: str
    brief: Brief
    brand: BrandProfile
    plan: Plan | None
    examples: list[Example]
    variants: list[Variant]
    critiques: list[Critique]
    revision_count: int
    errors: Annotated[list[RunError], add_errors]
    usage: Annotated[Usage, add_usage]
    status: RunStatus
    result: RunResult | None
    started_at: (
        float  # seconds since the epoch when the run began; the wall-clock budget runs from it
    )
    trace_id: str | None  # Langfuse trace of this run (BF-25); None when tracing is off


def new_run_state(
    brief: Brief,
    brand: BrandProfile,
    *,
    run_id: str | None = None,
    started_at: float | None = None,
    trace_id: str | None = None,
) -> RunState:
    """The initial state for a run: inputs filled in, everything else empty.

    `started_at` defaults to now. It is the one thing a node cannot recompute, which is why it
    lives in state: the router and the gateway both measure the wall-clock budget from it.
    """
    return RunState(
        run_id=run_id or uuid4().hex,
        brief=brief,
        brand=brand,
        plan=None,
        examples=[],
        variants=[],
        critiques=[],
        revision_count=0,
        errors=[],
        usage=Usage(),
        status="running",
        result=None,
        started_at=time.time() if started_at is None else started_at,
        trace_id=trace_id,
    )
