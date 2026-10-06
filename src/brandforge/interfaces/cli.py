"""Typer CLI: `brandforge generate --brand X --brief brief.yaml`.

`generate` runs the brief through the LangGraph pipeline (`brandforge.graph`).
"""

from pathlib import Path
from typing import Annotated, NoReturn

import typer
import yaml
from pydantic import ValidationError

from brandforge import __version__
from brandforge.brands import BrandLoadError, load_brand
from brandforge.graph import run_graph
from brandforge.llm.base import GatewayError
from brandforge.models import Brief, Usage, Variant

app = typer.Typer(
    help="Turn a creative brief into on-brand ad copy.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(show: bool) -> None:
    if show:
        typer.echo(f"brandforge {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """BrandForge command line."""


def load_brief(path: Path) -> Brief:
    """Read a brief YAML file and validate it into a `Brief`.

    Raises `ValueError` for anything wrong with the file, so the caller has one error to catch.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a brief must be a YAML mapping")
    try:
        return Brief.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def format_variants(variants: list[Variant]) -> str:
    blocks = [
        f"[{v.id}] {v.channel}\n  Headline: {v.headline}\n  Body:     {v.body}\n  CTA:      {v.cta}"
        for v in variants
    ]
    return "\n\n".join(blocks) if blocks else "(the model returned no variants)"


def format_usage(usage: Usage) -> str:
    return (
        f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out "
        f"({usage.total_tokens} total)\n"
        f"Cost:   ${usage.cost_usd:.4f}"
    )


def _fail(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def generate(
    brand: Annotated[str, typer.Option("--brand", help="Brand ID, e.g. voltride.")],
    brief: Annotated[
        Path,
        typer.Option(
            "--brief",
            help="Path to a brief YAML file.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """Generate ad copy variants for a brief and print them with token cost."""
    try:
        brief_model = load_brief(brief)
    except ValueError as exc:
        _fail(str(exc))

    try:
        brand_profile = load_brand(brand)
    except BrandLoadError as exc:
        _fail(str(exc))

    try:
        state = run_graph(brief_model, brand_profile)
    except (GatewayError, FileNotFoundError) as exc:
        _fail(f"generation failed: {exc}")

    typer.echo(format_variants(state["variants"]))
    typer.echo("")
    typer.echo(format_usage(state["usage"]))
