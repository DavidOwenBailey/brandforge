"""Load the fixed eval dataset (BF-33).

Each case is one YAML file in the repo's ``evals/cases/`` directory, next to
the promptfoo config. The file holds the case id, the brand id, the brief,
and any hard constraints a string check can decide. ``load_cases`` reads the
whole set.
"""

import re
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from brandforge.models import Brief, Channel, NonEmptyStr

# Channel word first, then "max N characters", so two caps in one line both count.
_CHANNEL_LIMIT = re.compile(
    r"\b(search|social|display|email)\b.*?\bmax\s+(\d+)\s+characters\b",
    re.IGNORECASE,
)
_CHANNELS: dict[str, Channel] = {
    "search": "search",
    "social": "social",
    "display": "display",
    "email": "email",
}
HeadlineLimit = Annotated[int, Field(ge=1)]


def _find_cases_dir() -> Path:
    """The checkout's ``evals/cases``, found by walking up to ``pyproject.toml``.

    An installed copy that is not a checkout falls back to ``evals/cases``
    relative to the working directory, which is the repo root for ``uv run``.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "brandforge").is_dir():
            return parent / "evals" / "cases"
    return Path("evals/cases")


CASES_DIR = _find_cases_dir()


class CaseLoadError(Exception):
    """An eval case is missing, malformed or fails validation."""


class HardConstraints(BaseModel):
    """Machine-checkable limits for one case.

    ``headline_max_chars`` caps the headline on named channels. A channel that
    is absent is not length-checked. ``must_mention`` lists phrases that must
    appear in the headline, body or call to action. Brand banned words stay on
    the brand profile and are not copied here.
    """

    model_config = ConfigDict(extra="forbid")

    headline_max_chars: dict[Channel, HeadlineLimit] = Field(default_factory=dict)
    must_mention: list[NonEmptyStr] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One offline eval row: which brand, which brief, which hard checks."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    brand_id: NonEmptyStr
    brief: Brief
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)

    @model_validator(mode="after")
    def _hard_constraints_match_the_brief(self) -> "EvalCase":
        stated = _stated_headline_limits(self.brief.constraints)
        recorded = dict(self.hard_constraints.headline_max_chars)
        requested = set(self.brief.channels)
        off_brief = (set(stated) | set(recorded)) - requested
        if off_brief:
            names = ", ".join(sorted(off_brief))
            raise ValueError(f"headline limits name channels the brief does not request: {names}")
        if stated != recorded:
            raise ValueError(
                "headline_max_chars must equal the caps named in the brief constraints. "
                "Write each cap as '<channel> ... max <n> characters'. "
                f"Constraints state {_fmt_limits(stated)}; "
                f"hard_constraints has {_fmt_limits(recorded)}."
            )
        blob = " ".join(self.brief.constraints).casefold()
        for phrase in self.hard_constraints.must_mention:
            if phrase.casefold() not in blob:
                raise ValueError(f"must_mention {phrase!r} is not stated in the brief constraints")
        return self


def _stated_headline_limits(constraints: list[str]) -> dict[Channel, int]:
    """Caps written in the brief, keyed by channel.

    The channel word has to come before ``max N characters``. Two different
    numbers for one channel are an error.
    """
    found: dict[Channel, int] = {}
    for text in constraints:
        for match in _CHANNEL_LIMIT.finditer(text):
            channel = _CHANNELS[match.group(1).lower()]
            limit = int(match.group(2))
            previous = found.get(channel)
            if previous is not None and previous != limit:
                raise ValueError(
                    f"constraints disagree on the {channel} headline limit ({previous} and {limit})"
                )
            found[channel] = limit
    return found


def _fmt_limits(limits: dict[Channel, int]) -> str:
    if not limits:
        return "{}"
    parts = ", ".join(f"{name}={limits[name]}" for name in sorted(limits))
    return f"{{{parts}}}"


def list_case_ids(cases_dir: Path = CASES_DIR) -> list[str]:
    return sorted(p.stem for p in cases_dir.glob("*.yaml"))


def load_case(case_id: str, cases_dir: Path = CASES_DIR) -> EvalCase:
    """Load and validate one case.

    Raises `CaseLoadError` for an unknown id, invalid YAML, a schema violation,
    an id that does not match the filename, or a filename that does not start
    with ``<brand_id>_``. Checking the id against the directory listing also
    blocks path tricks such as ``../x``.
    """
    if case_id not in list_case_ids(cases_dir):
        known = ", ".join(list_case_ids(cases_dir)) or "(none)"
        raise CaseLoadError(f"Unknown case '{case_id}'. Known cases: {known}")

    path = cases_dir / f"{case_id}.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CaseLoadError(f"{path.name}: invalid YAML: {exc}") from exc

    try:
        case = EvalCase.model_validate(raw)
    except ValidationError as exc:
        raise CaseLoadError(f"{path.name}: {exc}") from exc

    if case.id != path.stem:
        raise CaseLoadError(f"{path.name}: id '{case.id}' must match the filename")
    if not path.stem.startswith(f"{case.brand_id}_"):
        raise CaseLoadError(
            f"{path.name}: filename must start with the brand id '{case.brand_id}_'"
        )
    return case


def load_cases(cases_dir: Path = CASES_DIR) -> list[EvalCase]:
    """Load every case in filename order."""
    return [load_case(case_id, cases_dir) for case_id in list_case_ids(cases_dir)]
