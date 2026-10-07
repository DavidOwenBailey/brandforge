"""Brand critic agent: scores each variant against the brand's rubric (BF-15).

The critic makes one strong-tier call per variant, so each variant is scored on its own, with
no neighbouring copy to compare against. The model returns a `CriticReply`; `build_critique`
turns it into a `Critique` and sets `passed` from the configured thresholds, so the model never
decides pass or fail. Variant ids come from state, never from the model.
"""

from typing import Any, Protocol

from brandforge.config import Settings, Tier, get_settings
from brandforge.llm.gateway import complete_structured
from brandforge.models import BrandProfile, Brief, Critique, RunState, Usage, Variant
from brandforge.prompts.loader import load_prompt, render_prompt
from brandforge.scoring import CriticReply, build_critique

CRITIC_TIER: Tier = "strong"  # judging copy needs judgement; the planner uses the same tier


class StructuredCompleter(Protocol):
    """The slice of `complete_structured` this module uses. Tests pass a fake."""

    def __call__(
        self,
        prompt: str,
        schema: type[CriticReply],
        tier: Tier,
        /,
        *,
        settings: Settings | None = None,
    ) -> tuple[CriticReply, Usage]: ...


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def _rubric_text(brand: BrandProfile) -> str:
    """Every criterion with all five anchors, best first. The critic needs the full scale."""
    blocks = []
    for criterion in brand.rubric.criteria:
        levels = "\n".join(f"  {level}: {criterion.anchors[level]}" for level in range(5, 0, -1))
        blocks.append(f"- {criterion.name}: {criterion.description}\n{levels}")
    return "\n".join(blocks)


def build_critic_prompt(
    brief: Brief, brand: BrandProfile, variant: Variant, *, version: str
) -> str:
    return render_prompt(
        load_prompt("critic", version),
        brand_id=brand.id,
        voice=", ".join(brand.voice),
        do=_bullets(brand.do),
        dont=_bullets(brand.dont),
        banned_words=", ".join(brand.banned_words) or "(none)",
        criteria_names=", ".join(c.name for c in brand.rubric.criteria),
        rubric=_rubric_text(brand),
        channel=variant.channel,
        headline=variant.headline,
        body=variant.body,
        cta=variant.cta,
        product=brief.product,
        objective=brief.objective,
        constraints=_bullets(brief.constraints),
    )


def critique_variants(
    state: RunState,
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter = complete_structured,
) -> dict[str, Any]:
    """Graph node: reads `brief`, `brand` and `variants`; returns `critiques` and `usage`.

    Returns one `Critique` per variant, in variant order, replacing any earlier critiques. With
    no variants it makes no call. Raises `CritiqueError` when a reply does not fit the rubric,
    and whatever the gateway raises; handling those failures is the graph's job (BF-22).
    """
    cfg = settings or get_settings()
    brief, brand = state["brief"], state["brand"]

    critiques: list[Critique] = []
    usage = Usage()
    for variant in state["variants"]:
        prompt = build_critic_prompt(brief, brand, variant, version=cfg.critic_prompt_version)
        reply, call_usage = complete(prompt, CriticReply, CRITIC_TIER, settings=cfg)
        usage = usage + call_usage
        critiques.append(build_critique(variant.id, reply, brand.rubric, cfg.thresholds))

    return {"critiques": critiques, "usage": usage}
