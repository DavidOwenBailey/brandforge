"""Reviser agent: rewrites only the failing variants, using the critic's notes (BF-17).

The reviser makes one fast-tier call per failing variant. Passing variants are returned
untouched and in their original order, so a revision cannot make good copy worse. Ids and
channels come from state, never from the model, so a rewrite replaces its variant in place.
Each pass counts as one revision: `revision_count` goes up by one, which is what the router
checks to keep the loop bounded.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from brandforge.config import Settings, Tier, get_settings
from brandforge.llm.gateway import complete_structured
from brandforge.models import BrandProfile, Brief, Critique, NonEmptyStr, RunState, Usage, Variant
from brandforge.prompts.loader import load_prompt, render_prompt
from brandforge.router import failing_variant_ids

REVISER_TIER: Tier = "fast"  # the reviser runs as often as the writer, so it uses the same tier


class ReviserReply(BaseModel):
    """What the model returns for one rewritten variant. Id and channel are set in code."""

    model_config = ConfigDict(extra="forbid")

    headline: NonEmptyStr
    body: str
    cta: NonEmptyStr


class StructuredCompleter(Protocol):
    """The slice of `complete_structured` this module uses. Tests pass a fake."""

    def __call__(
        self,
        prompt: str,
        schema: type[ReviserReply],
        tier: Tier,
        /,
        *,
        settings: Settings | None = None,
    ) -> tuple[ReviserReply, Usage]: ...


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def _rubric_text(brand: BrandProfile) -> str:
    return "\n".join(
        f"- {c.name}: {c.description} A 5 looks like: {c.anchors[5]}" for c in brand.rubric.criteria
    )


def _critique_text(critique: Critique | None) -> str:
    """The critic's scores and fixes for one variant, as prompt text.

    A critique can fail a variant yet list no fixes (see ADR 0012), and a variant can have no
    critique at all (the router counts that as failing). Both still get a rewrite, with an
    instruction that does not depend on notes that are not there.
    """
    if critique is None:
        return "The critic has not scored this variant. Improve it against the criteria above."
    scores = "\n".join(f"- {name}: {score}" for name, score in critique.scores.items())
    if critique.fixes:
        fixes = _bullets(critique.fixes)
    else:
        fixes = "- (The critic gave no specific fixes. Raise the lowest-scoring criteria.)"
    return f"Scores (1 to 5):\n{scores}\nFixes:\n{fixes}"


def build_reviser_prompt(
    brief: Brief,
    brand: BrandProfile,
    variant: Variant,
    critique: Critique | None,
    *,
    version: str,
) -> str:
    return render_prompt(
        load_prompt("reviser", version),
        brand_id=brand.id,
        voice=", ".join(brand.voice),
        do=_bullets(brand.do),
        dont=_bullets(brand.dont),
        banned_words=", ".join(brand.banned_words) or "(none)",
        rubric=_rubric_text(brand),
        channel=variant.channel,
        headline=variant.headline,
        body=variant.body,
        cta=variant.cta,
        critique=_critique_text(critique),
        product=brief.product,
        objective=brief.objective,
        constraints=_bullets(brief.constraints),
    )


def revise_variants(
    state: RunState,
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter = complete_structured,
) -> dict[str, Any]:
    """Graph node: reads `brief`, `brand`, `variants`, `critiques` and `revision_count`.

    Returns `variants` (failing ones rewritten in place, the rest unchanged), `critiques`
    (without the ones for rewritten variants, which no longer describe the new copy),
    `revision_count` (one higher) and `usage`. If nothing is failing it makes no call and
    returns an empty update, without counting a revision. Raises whatever the gateway raises;
    handling that is the graph's job (BF-22).
    """
    cfg = settings or get_settings()
    brief, brand = state["brief"], state["brand"]

    failing = set(failing_variant_ids(state))
    if not failing:
        return {}

    critiques = {critique.variant_id: critique for critique in state["critiques"]}
    variants: list[Variant] = []
    usage = Usage()
    for variant in state["variants"]:
        if variant.id not in failing:
            variants.append(variant)
            continue
        prompt = build_reviser_prompt(
            brief, brand, variant, critiques.get(variant.id), version=cfg.reviser_prompt_version
        )
        reply, call_usage = complete(prompt, ReviserReply, REVISER_TIER, settings=cfg)
        usage = usage + call_usage
        variants.append(
            Variant(
                id=variant.id,
                channel=variant.channel,
                headline=reply.headline,
                body=reply.body,
                cta=reply.cta,
            )
        )

    return {
        "variants": variants,
        "critiques": [c for c in state["critiques"] if c.variant_id not in failing],
        "revision_count": state["revision_count"] + 1,
        "usage": usage,
    }
