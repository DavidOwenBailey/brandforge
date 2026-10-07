"""Planner agent: turns a brief and a brand profile into a structured `Plan` (BF-13).

The planner decides the audience wording, the angle and how many variants to write per
channel. It does not decide the channels: those are the brief's, set in code, so the
plan can never drop or invent one.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from brandforge.config import Settings, Tier, get_settings
from brandforge.llm.gateway import complete_structured
from brandforge.models import BrandProfile, Brief, NonEmptyStr, Plan, RunState, Usage
from brandforge.prompts.loader import load_prompt, render_prompt

PLANNER_TIER: Tier = "strong"  # planning needs judgement; it runs once per run


class PlanReply(BaseModel):
    """What the model returns. Channels are not asked for: they come from the brief.

    `variants_per_channel` has no range here because strict structured-output modes do
    not all support numeric bounds; the planner clamps it in code instead.
    """

    model_config = ConfigDict(extra="forbid")

    audience: NonEmptyStr
    angle: NonEmptyStr
    variants_per_channel: int


class StructuredCompleter(Protocol):
    """The slice of `complete_structured` this module uses. Tests pass a fake."""

    def __call__(
        self,
        prompt: str,
        schema: type[PlanReply],
        tier: Tier,
        /,
        *,
        settings: Settings | None = None,
    ) -> tuple[PlanReply, Usage]: ...


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def build_planner_prompt(
    brief: Brief, brand: BrandProfile, *, version: str, max_variants_per_channel: int
) -> str:
    return render_prompt(
        load_prompt("planner", version),
        brand_id=brand.id,
        voice=", ".join(brand.voice),
        do=_bullets(brand.do),
        dont=_bullets(brand.dont),
        banned_words=", ".join(brand.banned_words) or "(none)",
        max_variants_per_channel=str(max_variants_per_channel),
        channels=", ".join(brief.channels),
        product=brief.product,
        audience=brief.audience,
        objective=brief.objective,
        constraints=_bullets(brief.constraints),
    )


def create_plan(
    brief: Brief,
    brand: BrandProfile,
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter = complete_structured,
) -> tuple[Plan, Usage]:
    """Return the plan for a brief and the usage of the one call that produced it.

    Raises whatever the gateway raises (for example `StructuredOutputError`); handling
    failures is the graph's job (BF-22).
    """
    cfg = settings or get_settings()
    max_variants = cfg.planner_max_variants_per_channel
    prompt = build_planner_prompt(
        brief,
        brand,
        version=cfg.planner_prompt_version,
        max_variants_per_channel=max_variants,
    )
    reply, usage = complete(prompt, PlanReply, PLANNER_TIER, settings=cfg)
    plan = Plan(
        audience=reply.audience,
        angle=reply.angle,
        channels=list(brief.channels),
        variants_per_channel=min(max(reply.variants_per_channel, 1), max_variants),
    )
    return plan, usage


def plan_brief(
    state: RunState,
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter = complete_structured,
) -> dict[str, Any]:
    """Graph node: reads `brief` and `brand`, returns only `plan` and this call's `usage`."""
    plan, usage = create_plan(state["brief"], state["brand"], settings=settings, complete=complete)
    return {"plan": plan, "usage": usage}
