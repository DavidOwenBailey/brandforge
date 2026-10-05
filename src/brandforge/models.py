"""Pydantic data contracts for every boundary in BrandForge."""

from typing import Annotated, Literal

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
        # Used later as the reducer that accumulates usage across nodes.
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )
