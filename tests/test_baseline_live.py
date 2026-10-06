"""Live check of the baseline against a real provider. Opt-in; never runs in CI.

Run from the repo root with a Gemini fast tier, for example (PowerShell):

    $env:BRANDFORGE_LIVE = "1"
    $env:BRANDFORGE_MODELS__FAST = "gemini:<model-id>"
    $env:BRANDFORGE_PRICING__FAST__INPUT_PER_MTOK = "<price>"
    $env:BRANDFORGE_PRICING__FAST__OUTPUT_PER_MTOK = "<price>"
    uv run pytest tests/test_baseline_live.py -s

GEMINI_API_KEY is read from .env.
"""

import os

import pytest

from brandforge.baseline import generate_baseline
from brandforge.brands.loader import load_brand
from brandforge.config import get_settings
from brandforge.models import Brief

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("BRANDFORGE_LIVE") != "1", reason="set BRANDFORGE_LIVE=1"),
]


def test_baseline_live_on_gemini_fast_tier() -> None:
    settings = get_settings()
    assert settings.models.resolve("fast").provider == "gemini", (
        "Set BRANDFORGE_MODELS__FAST=gemini:<model-id> for this test"
    )
    brand = load_brand("voltride")
    brief = Brief(
        product="Voltride commuter e-bike",
        audience="city commuters",
        objective="conversion",
        channels=["search", "social"],
        constraints=["no discounts"],
    )

    variants, usage = generate_baseline(brief, brand, settings=settings)

    print(f"\n{len(variants)} variants, usage: {usage}")
    for v in variants:
        print(f"[{v.channel}] {v.headline} | {v.body} | {v.cta}")
    assert variants
    assert usage.input_tokens > 0 and usage.output_tokens > 0
    assert {v.channel for v in variants} <= set(brief.channels)
    banned = [w.lower() for w in brand.banned_words]
    for v in variants:
        text = f"{v.headline} {v.body} {v.cta}".lower()
        assert not [w for w in banned if w in text], f"banned word in: {text}"
