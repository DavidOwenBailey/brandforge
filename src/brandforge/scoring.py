"""The critic's reply contract and its conversion into a `Critique` (BF-15).

Strict structured-output modes cannot express a free-form dict, so the model never returns
`Critique.scores` directly. It returns a `CriticReply`, a list of `{criterion, score}` items,
and `build_critique` turns that into a `Critique` in code: it checks that every rubric
criterion is scored exactly once, that each score is 1 to 5, and computes `overall` and
`passed` from the configured thresholds. The model never decides whether a variant passes.
"""

from pydantic import BaseModel, ConfigDict

from brandforge.config import Thresholds
from brandforge.models import Critique, NonEmptyStr, Rubric


class CritiqueError(ValueError):
    """The critic's reply does not fit the brand's rubric, so no `Critique` can be built."""


class CriterionScore(BaseModel):
    """One criterion's score. `score` has no range here: not every provider's strict mode
    supports numeric bounds, so `build_critique` checks it instead."""

    model_config = ConfigDict(extra="forbid")

    criterion: NonEmptyStr
    score: int


class CriticReply(BaseModel):
    """What the model returns for one variant. Lists of items, never a dict, so the schema
    stays portable across providers. The variant id is not asked for: it is set in code."""

    model_config = ConfigDict(extra="forbid")

    scores: list[CriterionScore]
    fixes: list[NonEmptyStr]


def build_critique(
    variant_id: str, reply: CriticReply, rubric: Rubric, thresholds: Thresholds
) -> Critique:
    """Convert a `CriticReply` into a `Critique` for `variant_id`.

    `scores` follows the rubric's order. `overall` is the mean of the criterion scores. The
    variant passes when `overall` is at least `thresholds.min_overall` and no criterion is
    below `thresholds.min_per_criterion`.

    Raises:
        CritiqueError: a rubric criterion is missing or scored twice, the reply scores a
            criterion the rubric does not have, or a score is outside 1 to 5.
    """
    expected = [criterion.name for criterion in rubric.criteria]
    given = [item.criterion for item in reply.scores]

    problems: list[str] = []
    missing = [name for name in expected if name not in given]
    if missing:
        problems.append(f"missing criteria: {', '.join(missing)}")
    unknown = sorted({name for name in given if name not in expected})
    if unknown:
        problems.append(f"unknown criteria: {', '.join(unknown)}")
    repeated = sorted({name for name in given if given.count(name) > 1})
    if repeated:
        problems.append(f"criteria scored more than once: {', '.join(repeated)}")
    out_of_range = [
        f"{item.criterion}={item.score}" for item in reply.scores if not 1 <= item.score <= 5
    ]
    if out_of_range:
        problems.append(f"scores outside 1 to 5: {', '.join(out_of_range)}")
    if problems:
        raise CritiqueError(
            f"Critic reply for variant {variant_id!r} is unusable: " + "; ".join(problems)
        )

    by_name = {item.criterion: item.score for item in reply.scores}
    scores = {name: by_name[name] for name in expected}
    overall = sum(scores.values()) / len(scores)
    passed = (
        overall >= thresholds.min_overall and min(scores.values()) >= thresholds.min_per_criterion
    )
    return Critique(
        variant_id=variant_id,
        scores=scores,
        overall=overall,
        passed=passed,
        fixes=list(reply.fixes),
    )
