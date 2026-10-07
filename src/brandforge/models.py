"""Pydantic data contracts for every boundary in BrandForge."""

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


class Usage(_Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        # The reducer that accumulates usage across nodes (see `RunState`).
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class Plan(_Contract):
    """The planner's structured reading of a brief (written by the planner, BF-13)."""

    audience: NonEmptyStr
    angle: NonEmptyStr
    channels: list[Channel] = Field(min_length=1)
    variants_per_channel: int = Field(ge=1, le=10)


class Example(_Contract):
    """An approved piece of copy retrieved as a style reference (written by BF-31)."""

    brand_id: NonEmptyStr
    channel: Channel
    headline: NonEmptyStr
    body: str
    cta: NonEmptyStr


class RunError(_Contract):
    """A failure recorded in state so the run can still end in a defined status."""

    node: NonEmptyStr
    message: NonEmptyStr


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
    that produced it. The trace ID joins it when tracing lands (BF-25).
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


def new_run_state(brief: Brief, brand: BrandProfile, *, run_id: str | None = None) -> RunState:
    """The initial state for a run: inputs filled in, everything else empty."""
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
    )
