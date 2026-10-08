"""Single-prompt baseline: one structured call, brief and brand profile in, variants out.

This is the system the full pipeline is measured against (see the evaluation design),
so it deliberately gets no extra scaffolding: one prompt, the same tier as the writer.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from brandforge.config import Settings, Tier, get_settings
from brandforge.llm.gateway import complete_structured
from brandforge.models import BrandProfile, Brief, Channel, NonEmptyStr, Usage, Variant
from brandforge.prompts.loader import PromptParts, load_prompt, render_prompt_parts

BASELINE_TIER: Tier = "fast"  # same tier as the writer, so the comparison is fair


class VariantDraft(BaseModel):
    """What the model writes. Ids are assigned in code, never by the model."""

    model_config = ConfigDict(extra="forbid")

    channel: Channel
    headline: NonEmptyStr
    body: str
    cta: NonEmptyStr


class BaselineReply(BaseModel):
    """A list of items, never a dict, so the schema stays portable across providers."""

    model_config = ConfigDict(extra="forbid")

    variants: list[VariantDraft]


class StructuredCompleter(Protocol):
    """The slice of `complete_structured` this module uses. Tests pass a fake."""

    def __call__(
        self,
        prompt: str,
        schema: type[BaselineReply],
        tier: Tier,
        /,
        *,
        system: str | None = None,
        settings: Settings | None = None,
    ) -> tuple[BaselineReply, Usage]: ...


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def baseline_prompt(
    brief: Brief, brand: BrandProfile, *, version: str, variants_per_channel: int
) -> PromptParts:
    """The baseline prompt. The brand profile is the cached system prefix (BF-27)."""
    return render_prompt_parts(
        load_prompt("baseline", version),
        brand_id=brand.id,
        voice=", ".join(brand.voice),
        do=_bullets(brand.do),
        dont=_bullets(brand.dont),
        banned_words=", ".join(brand.banned_words) or "(none)",
        variants_per_channel=str(variants_per_channel),
        channels=", ".join(brief.channels),
        product=brief.product,
        audience=brief.audience,
        objective=brief.objective,
        constraints=_bullets(brief.constraints),
    )


def build_baseline_prompt(
    brief: Brief, brand: BrandProfile, *, version: str, variants_per_channel: int
) -> str:
    return baseline_prompt(
        brief, brand, version=version, variants_per_channel=variants_per_channel
    ).text


def generate_baseline(
    brief: Brief,
    brand: BrandProfile,
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter = complete_structured,
) -> tuple[list[Variant], Usage]:
    """Return the baseline's variants and the usage of the one call that produced them.

    Raises whatever the gateway raises (for example `StructuredOutputError`); handling
    failures is the caller's job.
    """
    cfg = settings or get_settings()
    prompt = baseline_prompt(
        brief,
        brand,
        version=cfg.baseline_prompt_version,
        variants_per_channel=cfg.baseline_variants_per_channel,
    )
    reply, usage = complete(
        prompt.user, BaselineReply, BASELINE_TIER, system=prompt.system, settings=cfg
    )
    variants = [
        Variant(
            id=f"baseline-{n}",
            channel=draft.channel,
            headline=draft.headline,
            body=draft.body,
            cta=draft.cta,
        )
        for n, draft in enumerate(reply.variants, start=1)
    ]
    return variants, usage
