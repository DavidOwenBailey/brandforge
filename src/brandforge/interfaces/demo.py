"""Demo-page client (BF-40).

The Streamlit page in ``app.py`` is a client of ``POST /generate``. Nothing here runs the
graph or calls a model. Brand ids come from the brand store, the same YAML the API loads.
A pasted brief is YAML or JSON, validated as a ``Brief``, then posted as JSON (ADR 0033).
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from pydantic import ValidationError

from brandforge.brands import list_brand_ids
from brandforge.config import Settings
from brandforge.models import Brief, RunResult, VariantResult


def brand_choices() -> list[str]:
    """Brand ids the page can offer, sorted, from the same YAML the API loads."""
    return list_brand_ids()


# The page Streamlit runs. `brandforge demo` points `streamlit run` at this file.
APP_PATH = Path(__file__).with_name("app.py")

# A run may start its last model call just before the wall-clock budget ends, and that
# call may use the full per-call timeout. The margin covers the assembler's reply.
_TIMEOUT_MARGIN_SECONDS = 15

# The text area starts with this brief so the page can be tried without hunting for a file.
# It is the Voltride commuter sample, and the brand select defaults to voltride to match.
EXAMPLE_BRIEF = """\
product: Voltride commuter e-bike
audience: city commuters
objective: conversion
channels: [search, social]
constraints:
  - no discounts
"""


class DemoError(Exception):
    """A brief or an API reply the page can show as written."""


@dataclass(frozen=True, slots=True)
class VariantView:
    """One variant prepared for the page: the copy, the scores and the flag."""

    variant_id: str
    channel: str
    headline: str
    body: str
    cta: str
    verdict: str  # "passed", "flagged", or "flagged (not scored)"
    overall: str  # one decimal, or "-" when the variant was never scored
    scores: tuple[tuple[str, str], ...]  # criterion, score, in rubric order
    fixes: tuple[str, ...]


def generation_timeout(settings: Settings) -> float:
    """How many seconds the page waits for one ``POST /generate``.

    The wait is the run's wall-clock budget, plus one full model-call timeout, plus a
    short margin. That covers a call that started just before the budget ran out.
    """
    return float(
        settings.budgets.max_wall_clock_seconds
        + settings.budgets.request_timeout_seconds
        + _TIMEOUT_MARGIN_SECONDS
    )


def parse_brief(text: str) -> Brief:
    """Read a pasted brief as YAML or JSON and validate it.

    A brief file's contents paste in unchanged. JSON is accepted because it is the
    shape the API uses, and it is valid YAML.
    """
    stripped = text.strip()
    if not stripped:
        raise DemoError("Paste a brief. It needs product, audience, objective and channels.")
    try:
        raw = yaml.safe_load(stripped)
    except yaml.YAMLError as exc:
        raise DemoError(f"The brief is not valid YAML or JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DemoError("A brief must be a YAML or JSON mapping.")
    try:
        return Brief.model_validate(raw)
    except ValidationError as exc:
        raise DemoError(_validation_message(exc)) from exc


def request_generation(
    brand_id: str,
    brief: Brief,
    *,
    api_base_url: str,
    timeout: float,
    client: httpx.Client | None = None,
) -> RunResult:
    """Post the brief to ``{api_base_url}/generate`` and return the run result.

    ``timeout`` is seconds and applies to this request even when ``client`` is passed.
    A finished run is returned for HTTP 200, including a run whose status is ``failed``.
    Anything else raises ``DemoError`` with a sentence the page can show.

    The caller owns ``client`` when it passes one. Otherwise this function opens a
    client and closes it.
    """
    base = api_base_url.strip().rstrip("/")
    url = f"{base}/generate"
    payload = {"brand": brand_id, "brief": brief.model_dump(mode="json")}
    http = client if client is not None else httpx.Client()
    try:
        try:
            response = http.post(url, json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise DemoError(
                f"The API at {base} did not respond in time. A run can take a couple of minutes."
            ) from exc
        except httpx.HTTPError as exc:
            raise DemoError(
                f"Cannot reach the API at {base}. Start it with `brandforge serve`."
            ) from exc
        if response.status_code == 200:
            return _parse_result(response)
        detail = _error_detail(response)
        if response.status_code == 404:
            raise DemoError(detail or "The API does not know that brand.")
        if response.status_code == 422:
            raise DemoError(detail or "The API rejected the brief.")
        raise DemoError(detail or f"The API returned HTTP {response.status_code}.")
    finally:
        if client is None:
            http.close()


def variant_views(result: RunResult) -> list[VariantView]:
    """The result's variants in display order, with scores and the flag filled in."""
    views: list[VariantView] = []
    for item in result.variants:
        critique = item.critique
        scores = (
            tuple((name, str(score)) for name, score in critique.scores.items())
            if critique is not None
            else ()
        )
        views.append(
            VariantView(
                variant_id=item.variant.id,
                channel=item.variant.channel,
                headline=item.variant.headline,
                body=item.variant.body,
                cta=item.variant.cta,
                verdict=_verdict(item),
                overall=f"{critique.overall:.1f}" if critique is not None else "-",
                scores=scores,
                fixes=tuple(critique.fixes) if critique is not None else (),
            )
        )
    return views


def score_rows(views: list[VariantView]) -> list[dict[str, str]]:
    """One row per variant. Criterion columns follow the order names first appear."""
    criteria: list[str] = []
    for view in views:
        for name, _score in view.scores:
            if name not in criteria:
                criteria.append(name)
    rows: list[dict[str, str]] = []
    for view in views:
        found = dict(view.scores)
        row = {"Variant": view.variant_id, "Channel": view.channel}
        for name in criteria:
            row[name] = found.get(name, "-")
        row["Overall"] = view.overall
        row["Result"] = view.verdict
        rows.append(row)
    return rows


def format_score_table(rows: list[dict[str, str]]) -> str:
    """A markdown table of score rows. Empty when there is nothing to show."""
    if not rows:
        return ""
    header = list(rows[0])

    def cell(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(name) for name in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row[name]) for name in header) + " |")
    return "\n".join(lines)


def format_variant(view: VariantView) -> str:
    """The copy for one variant, then its scores and any fixes still open."""
    lines = [
        f"**Headline:** {view.headline}",
        "",
        view.body,
        "",
        f"**CTA:** {view.cta}",
    ]
    if view.scores:
        scored = ", ".join(f"{name} {score}" for name, score in view.scores)
        lines.extend(["", f"**Scores:** {scored} (overall {view.overall})"])
    if view.fixes:
        lines.extend(["", "**Still to fix:** " + "; ".join(view.fixes)])
    return "\n".join(lines)


def status_text(result: RunResult) -> str:
    """The run status, how many variants came back, and how many are flagged."""
    return (
        f"Status: {result.status}. "
        f"{len(result.variants)} variants, {result.flagged_count} flagged. "
        f"Revisions: {result.revision_count}."
    )


def run_caption(result: RunResult) -> str:
    """Brand version, run id, trace id and cost, in one line."""
    trace = result.trace_id if result.trace_id else "tracing is off"
    return (
        f"Brand {result.brand_id} v{result.brand_version} "
        f"(rubric v{result.rubric_version})"
        f" · Run {result.run_id}"
        f" · Trace {trace}"
        f" · Cost ${result.usage.cost_usd:.4f}"
    )


def format_errors(result: RunResult) -> str:
    """The run's recorded errors, or an empty string when there are none."""
    if not result.errors:
        return ""
    lines = ["**Errors**", ""]
    lines.extend(f"- [{error.node}] {error.message}" for error in result.errors)
    return "\n".join(lines)


def streamlit_command(app_path: Path, *, host: str, port: int) -> list[str]:
    """The ``streamlit run`` command for the demo page.

    Headless skips the browser and the first-run email prompt. Usage stats stay off.
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def launch_demo(*, host: str, port: int, api_base_url: str) -> int:
    """Start the Streamlit page and return its exit code.

    The child inherits this process's environment. ``BRANDFORGE_API_BASE_URL`` is set to
    ``api_base_url``, and Streamlit's usage-stats flag is turned off.
    """
    env = os.environ.copy()
    env["BRANDFORGE_API_BASE_URL"] = api_base_url
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    command = streamlit_command(APP_PATH, host=host, port=port)
    return subprocess.call(command, env=env)


def _verdict(item: VariantResult) -> str:
    if not item.flagged:
        return "passed"
    if item.critique is not None:
        return "flagged"
    return "flagged (not scored)"


def _validation_message(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error["loc"])
        message = str(error["msg"])
        parts.append(f"{loc}: {message}" if loc else message)
    detail = " ".join(parts) if parts else "it does not match a brief"
    return f"The brief is not valid. {detail}"


def _parse_result(response: httpx.Response) -> RunResult:
    try:
        payload = response.json()
    except ValueError as exc:
        raise DemoError("The API returned a response that is not JSON.") from exc
    try:
        return RunResult.model_validate(payload)
    except ValidationError as exc:
        raise DemoError("The API returned a response that is not a run result.") from exc


def _error_detail(response: httpx.Response) -> str:
    """The API's ``detail`` as one sentence. Empty when the body is not that shape."""
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                loc = item.get("loc", [])
                where = ""
                if isinstance(loc, list):
                    where = ".".join(str(part) for part in loc if part != "body")
                message = str(item.get("msg", "invalid"))
                parts.append(f"{where}: {message}" if where else message)
            else:
                parts.append(str(item))
        return "; ".join(parts)
    return ""
