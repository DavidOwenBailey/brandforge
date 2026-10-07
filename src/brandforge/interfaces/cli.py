"""Typer CLI: `brandforge generate --brand X --brief brief.yaml`.

`generate` runs the brief through the LangGraph pipeline (`brandforge.graph`) and prints the
variants, a summary table of scores and flags, and the token cost.
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
from brandforge.models import Brief, RunResult, Usage, Variant

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
    return "\n\n".join(blocks) if blocks else "(no variants were produced)"


def _table(header: list[str], rows: list[list[str]]) -> str:
    """Left-aligned plain-text columns, two spaces apart, with a rule under the header."""
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return "  ".join(padded).rstrip()

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(header), rule, *(line(row) for row in rows)])


def format_summary(result: RunResult) -> str:
    """The run status, a score table with one row per variant, and what is left to fix.

    Columns are the criteria the critic scored, in the order they first appear. A variant that
    was never scored shows dashes. Flagged variants list the critic's remaining fixes below the
    table, and any errors recorded during the run close the summary.
    """
    lines = [
        f"Status:    {result.status}",
        f"Brand:     {result.brand_id} v{result.brand_version} (rubric v{result.rubric_version})",
        f"Variants:  {len(result.variants)} ({result.flagged_count} flagged)",
        f"Revisions: {result.revision_count}",
    ]

    criteria: list[str] = []
    for item in result.variants:
        if item.critique:
            criteria.extend(name for name in item.critique.scores if name not in criteria)

    rows: list[list[str]] = []
    for item in result.variants:
        critique = item.critique
        scores = [
            str(critique.scores[name]) if critique and name in critique.scores else "-"
            for name in criteria
        ]
        overall = f"{critique.overall:.1f}" if critique else "-"
        if not item.flagged:
            verdict = "passed"
        elif critique:
            verdict = "flagged"
        else:
            verdict = "flagged (not scored)"
        rows.append([item.variant.id, item.variant.channel, *scores, overall, verdict])
    if rows:
        header = ["Variant", "Channel", *criteria, "Overall", "Result"]
        lines += ["", _table(header, rows)]

    to_fix = [item for item in result.variants if item.flagged and item.critique]
    if any(item.critique and item.critique.fixes for item in to_fix):
        lines += ["", "Still to fix:"]
        for item in to_fix:
            if item.critique and item.critique.fixes:
                lines.append(f"  [{item.variant.id}] " + "; ".join(item.critique.fixes))

    if result.errors:
        lines += ["", "Errors:"]
        lines += [f"  [{error.node}] {error.message}" for error in result.errors]
    return "\n".join(lines)


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

    result = state["result"]
    if result is None:  # the graph always ends at the assembler, so this is a guard
        _fail("the run finished without a result")

    typer.echo(format_variants([item.variant for item in result.variants]))
    typer.echo("")
    typer.echo(format_summary(result))
    typer.echo("")
    typer.echo(format_usage(result.usage))
    if result.status == "failed":
        raise typer.Exit(code=1)
