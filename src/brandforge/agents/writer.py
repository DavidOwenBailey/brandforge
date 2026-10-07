"""Writer agent: turns a plan, a brand profile and optional examples into variants (BF-14).

The writer makes one call per channel in the plan. Channels, variant ids and the number of
variants kept are all set in code, so the model cannot add a channel, reuse an id or flood the
run with extra copy. Examples are optional: with none (the retriever arrives later, BF-31) the
prompt simply has no examples section.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from brandforge.config import Settings, Tier, get_settings
from brandforge.llm.gateway import complete_structured
from brandforge.models import (
    BrandProfile,
    Brief,
    Channel,
    Example,
    NonEmptyStr,
    Plan,
    RunError,
    RunState,
    Usage,
    Variant,
)
from brandforge.prompts.loader import load_prompt, render_prompt

WRITER_TIER: Tier = "fast"  # the writer runs most often; the baseline uses the same tier


class WriterDraft(BaseModel):
    """What the model writes for one variant. Channel and id are assigned in code."""

    model_config = ConfigDict(extra="forbid")

    headline: NonEmptyStr
    body: str
    cta: NonEmptyStr


class WriterReply(BaseModel):
    """A list of items, never a dict, so the schema stays portable across providers."""

    model_config = ConfigDict(extra="forbid")

    variants: list[WriterDraft]


class StructuredCompleter(Protocol):
    """The slice of `complete_structured` this module uses. Tests pass a fake."""

    def __call__(
        self,
        prompt: str,
        schema: type[WriterReply],
        tier: Tier,
        /,
        *,
        settings: Settings | None = None,
    ) -> tuple[WriterReply, Usage]: ...


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def _rubric_text(brand: BrandProfile) -> str:
    return "\n".join(
        f"- {c.name}: {c.description} A 5 looks like: {c.anchors[5]}" for c in brand.rubric.criteria
    )


def _examples_section(examples: list[Example], channel: Channel) -> str:
    """The examples block for one channel, or "" when there are none to show."""
    relevant = [e for e in examples if e.channel == channel]
    if not relevant:
        return ""
    lines = "\n".join(f"- Headline: {e.headline} | Body: {e.body} | CTA: {e.cta}" for e in relevant)
    return (
        "\n## Approved examples\n"
        f"Approved {channel} copy from this brand. Learn the voice and rhythm from it. "
        "Do not copy its wording.\n"
        f"<examples>\n{lines}\n</examples>\n"
    )


def build_writer_prompt(
    brief: Brief,
    brand: BrandProfile,
    plan: Plan,
    channel: Channel,
    examples: list[Example],
    *,
    version: str,
) -> str:
    return render_prompt(
        load_prompt("writer", version),
        brand_id=brand.id,
        voice=", ".join(brand.voice),
        do=_bullets(brand.do),
        dont=_bullets(brand.dont),
        banned_words=", ".join(brand.banned_words) or "(none)",
        rubric=_rubric_text(brand),
        examples_section=_examples_section(examples, channel),
        variants_count=str(plan.variants_per_channel),
        channel=channel,
        plan_audience=plan.audience,
        plan_angle=plan.angle,
        product=brief.product,
        objective=brief.objective,
        constraints=_bullets(brief.constraints),
    )


def write_variants(
    state: RunState,
    *,
    settings: Settings | None = None,
    complete: StructuredCompleter = complete_structured,
) -> dict[str, Any]:
    """Graph node: reads `brief`, `brand`, `plan` and `examples`; returns `variants` and `usage`.

    Adds an `errors` entry for any channel that came back with fewer variants than the plan
    asked for, and keeps what it did get. Raises `ValueError` if there is no plan, and whatever
    the gateway raises; handling those failures is the graph's job (BF-22).
    """
    plan = state["plan"]
    if plan is None:
        raise ValueError("The writer needs a plan; run the planner first.")
    cfg = settings or get_settings()

    variants: list[Variant] = []
    errors: list[RunError] = []
    usage = Usage()
    for channel in plan.channels:
        prompt = build_writer_prompt(
            state["brief"],
            state["brand"],
            plan,
            channel,
            state["examples"],
            version=cfg.writer_prompt_version,
        )
        reply, call_usage = complete(prompt, WriterReply, WRITER_TIER, settings=cfg)
        usage = usage + call_usage

        drafts = reply.variants[: plan.variants_per_channel]
        if len(drafts) < plan.variants_per_channel:
            errors.append(
                RunError(
                    node="writer",
                    message=(
                        f"channel {channel!r}: asked for {plan.variants_per_channel} "
                        f"variants, got {len(drafts)}"
                    ),
                )
            )
        variants.extend(
            Variant(
                id=f"{channel}-{n}",
                channel=channel,
                headline=draft.headline,
                body=draft.body,
                cta=draft.cta,
            )
            for n, draft in enumerate(drafts, start=1)
        )

    update: dict[str, Any] = {"variants": variants, "usage": usage}
    if errors:
        update["errors"] = errors
    return update
