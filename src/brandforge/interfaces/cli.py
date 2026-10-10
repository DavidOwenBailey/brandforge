"""Typer CLI: `brandforge generate`, `brandforge index`, `brandforge inspect <run_id>`,
`brandforge calibrate` and `brandforge eval-summary`.

`generate` runs the brief through the LangGraph pipeline (`brandforge.graph`) and prints the
variants, a summary table of scores and flags, the token cost broken down by node (BF-27), the
run ID and the Langfuse trace ID (BF-25). The state is checkpointed after every node (BF-24),
and `inspect` reads it back by run ID. `index` embeds the approved examples into one persistent
Chroma collection per brand (BF-30). `calibrate` compares the hand scores in `evals/calibration/`
with the judge (BF-37). `eval-summary` reads a promptfoo JSON export and prints the baseline
against the pipeline (BF-38). It does not call a model.

Logs are separate from that output (BF-26): JSON lines on stderr, each carrying the run ID while
a run is in progress. They are configured once, when the CLI starts.
"""

import json
import sqlite3
from pathlib import Path
from typing import Annotated, NoReturn

import typer
import yaml
from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from brandforge import __version__
from brandforge.brands import BrandLoadError, load_brand
from brandforge.checkpointing import RunStep, load_run_steps, open_checkpointer
from brandforge.config import get_settings
from brandforge.evals.calibration import CalibrationError, format_report, run_calibration
from brandforge.evals.results import ResultsError, summarize_path, write_summary
from brandforge.evals.results import format_summary as format_eval_summary
from brandforge.graph import run_graph
from brandforge.llm.base import GatewayError
from brandforge.logging import configure_logging
from brandforge.models import Brief, RunResult, Usage, Variant
from brandforge.retrieval import ExampleLoadError
from brandforge.retrieval.index import IndexBuildError, IndexedBrand, index_examples

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
    configure_logging(get_settings())


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


def _token_counts(usage: Usage) -> str:
    """The token counts as one phrase. Cache counts are omitted when a run did not use them."""
    parts = [f"{usage.input_tokens} in", f"{usage.output_tokens} out"]
    if usage.cache_write_tokens:
        parts.append(f"{usage.cache_write_tokens} cache write")
    if usage.cache_read_tokens:
        parts.append(f"{usage.cache_read_tokens} cache read")
    return ", ".join(parts)


def format_usage(usage: Usage) -> str:
    """The run's tokens and cost, then one row per node that spent anything.

    Nodes are in the order they first ran. A node that ran twice (the critic, after a revision)
    is one row, the sum of both visits.
    """
    lines = [
        f"Tokens: {_token_counts(usage)} ({usage.total_tokens} total)",
        f"Cost:   ${usage.cost_usd:.4f}",
    ]
    if not usage.nodes:
        return "\n".join(lines)
    show_cache = any(item.cache_write_tokens or item.cache_read_tokens for item in usage.nodes)
    header = ["Node", "Tokens", "Cost"]
    rows = [[item.node, str(item.total_tokens), f"${item.cost_usd:.4f}"] for item in usage.nodes]
    if show_cache:
        header = ["Node", "Tokens", "Cache write", "Cache read", "Cost"]
        rows = [
            [
                item.node,
                str(item.total_tokens),
                str(item.cache_write_tokens),
                str(item.cache_read_tokens),
                f"${item.cost_usd:.4f}",
            ]
            for item in usage.nodes
        ]
    lines += ["", "Cost by node:", _table(header, rows)]
    return "\n".join(lines)


def format_index_result(indexed: IndexedBrand) -> str:
    """One line for a brand that was written into the persistent index."""
    return (
        f"Indexed {indexed.count} examples for {indexed.brand_id} "
        f"into collection {indexed.collection!r} at {indexed.persist_dir}"
    )


def format_run_steps(run_id: str, steps: list[RunStep]) -> str:
    """One row per checkpoint: the state of the run after each node, oldest first.

    `After` is the node that just ran ("(input)" for the starting state) and `Next` is what was
    due to run, so a run that was cut short ends on a row that still has a `Next`. Counts and
    totals are cumulative, as they are in state. Errors recorded by the end of the run are
    listed under the table.
    """
    rows: list[list[str]] = []
    for step in steps:
        state = step.state
        critiques = state["critiques"]
        usage = state["usage"]
        rows.append(
            [
                str(step.step),
                step.node or "(input)",
                str(len(state["variants"])),
                f"{sum(1 for c in critiques if c.passed)}/{len(critiques)}",
                str(state["revision_count"]),
                str(len(state["errors"])),
                str(usage.total_tokens),
                f"${usage.cost_usd:.4f}",
                state["status"],
                ", ".join(step.next_nodes) or "-",
            ]
        )
    header = [
        "Step",
        "After",
        "Variants",
        "Passed",
        "Revisions",
        "Errors",
        "Tokens",
        "Cost",
        "Status",
        "Next",
    ]
    lines = [f"Run {run_id}: {len(steps)} checkpoints", "", _table(header, rows)]
    if steps[-1].state["errors"]:
        lines += ["", "Errors:"]
        lines += [f"  [{error.node}] {error.message}" for error in steps[-1].state["errors"]]
    return "\n".join(lines)


def format_step_state(step: RunStep) -> str:
    """The whole state after one step as indented JSON."""
    return json.dumps(to_jsonable_python(step.state), indent=2)


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
        with open_checkpointer(get_settings().checkpoint_db) as checkpointer:
            state = run_graph(brief_model, brand_profile, checkpointer=checkpointer)
    except (GatewayError, OSError, sqlite3.Error) as exc:
        _fail(f"generation failed: {exc}")

    result = state["result"]
    if result is None:  # the graph always ends at the assembler, so this is a guard
        _fail("the run finished without a result")

    typer.echo(format_variants([item.variant for item in result.variants]))
    typer.echo("")
    typer.echo(format_summary(result))
    typer.echo("")
    typer.echo(format_usage(result.usage))
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Trace ID: {result.trace_id}" if result.trace_id else "Trace ID: (tracing is off)")
    if result.status == "failed":
        raise typer.Exit(code=1)


@app.command("index")
def index_brands(
    brand: Annotated[
        str | None,
        typer.Option(
            "--brand",
            help="Brand to index. Defaults to every brand with approved examples.",
        ),
    ] = None,
) -> None:
    """Embed approved examples into a persistent Chroma collection per brand."""
    try:
        indexed = index_examples(
            None if brand is None else [brand],
            persist_dir=get_settings().chroma_dir,
        )
    except (ExampleLoadError, IndexBuildError, OSError) as exc:
        _fail(str(exc))

    if not indexed:
        _fail("no example brands to index")
    for item in indexed:
        typer.echo(format_index_result(item))


@app.command("inspect")
def inspect_run(
    run_id: Annotated[str, typer.Argument(help="The run ID that `generate` printed.")],
    step: Annotated[
        int | None,
        typer.Option("--step", help="Print the full state after this step, as JSON."),
    ] = None,
) -> None:
    """Show the state of a run after each node, from its checkpoints."""
    path = get_settings().checkpoint_db
    try:
        with open_checkpointer(path, create=False) as checkpointer:
            steps = load_run_steps(checkpointer, run_id)
    except FileNotFoundError as exc:
        _fail(f"{exc}; run `brandforge generate` first")
    except (OSError, sqlite3.Error) as exc:
        _fail(f"cannot read {path}: {exc}")

    if not steps:
        _fail(f"no checkpoints for run {run_id!r} in {path}")

    if step is None:
        typer.echo(format_run_steps(run_id, steps))
        return
    match = next((item for item in steps if item.step == step), None)
    if match is None:
        available = ", ".join(str(item.step) for item in steps)
        _fail(f"run {run_id!r} has no step {step} (steps: {available})")
    typer.echo(format_step_state(match))


@app.command("calibrate")
def calibrate_judge() -> None:
    """Compare hand scores in evals/calibration/ with the judge.

    Calls the judge once per authored output and prints quadratic weighted kappa.
    Kappa below 0.60 still exits 0: it is a finding about the rubric. Status 1 means
    the sample cannot be read, or the judge scored nothing.
    """
    try:
        report = run_calibration()
    except CalibrationError as exc:
        _fail(str(exc))
    typer.echo(format_report(report))
    if report.scored == 0:
        raise typer.Exit(code=1)


@app.command("eval-summary")
def eval_summary(
    export: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="promptfoo JSON export from --output.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="File stem to write. evals/results/full writes full.md and full.json.",
        ),
    ] = None,
) -> None:
    """Summarize a promptfoo export: the baseline beside the pipeline.

    Prints judge means, pass rate, deterministic checks, cost and latency.
    With --output, also writes that report as markdown and JSON. Does not call a model.
    """
    try:
        summary = summarize_path(export, settings=get_settings())
    except ResultsError as exc:
        _fail(str(exc))
    if output is not None:
        if output.is_dir():
            _fail(f"{output} is a directory. Pass a file stem, such as evals/results/full.")
        try:
            markdown_path, json_path = write_summary(summary, output)
        except OSError as exc:
            _fail(f"cannot write {output}: {exc}")
        typer.echo(f"Wrote {markdown_path}", err=True)
        typer.echo(f"Wrote {json_path}", err=True)
    typer.echo(format_eval_summary(summary))
