"""HTTP API (BF-39): ``POST /generate`` runs one brief and returns the result.

The handler is the CLI's generate path as HTTP. It loads the brand, runs the same graph
with the same checkpoint file, and returns the assembler's ``RunResult``. That result
already carries the Langfuse trace ID (BF-25). There is no second prompt and no second
pipeline.

The route has no auth. ``brandforge serve`` binds to loopback unless told otherwise
(ADR 0032). Interactive OpenAPI docs are at ``/docs``.
"""

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException
from fastapi.openapi.models import Example
from pydantic import BaseModel, ConfigDict, Field

from brandforge import __version__
from brandforge.brands import BrandLoadError, load_brand
from brandforge.checkpointing import open_checkpointer
from brandforge.config import get_settings
from brandforge.graph import run_graph
from brandforge.llm.base import GatewayError
from brandforge.logging import configure_logging
from brandforge.models import Brief, NonEmptyStr, RunResult

# Shown in the OpenAPI docs so a reader can try the route without inventing a brief.
_VOLTRIDE_EXAMPLE: Example = {
    "summary": "A Voltride commuter brief",
    "value": {
        "brand": "voltride",
        "brief": {
            "product": "Voltride commuter e-bike",
            "audience": "city commuters",
            "objective": "conversion",
            "channels": ["search", "social"],
            "constraints": ["no discounts"],
        },
    },
}


class GenerateRequest(BaseModel):
    """A brand id and the brief to run. The brief matches a brief YAML file."""

    model_config = ConfigDict(extra="forbid")

    brand: NonEmptyStr = Field(description="Brand id, for example voltride.")
    brief: Brief


class ErrorBody(BaseModel):
    """An HTTP error. A finished run is not one of these: its status is in the result."""

    detail: str


def run_generation(brand_id: str, brief: Brief) -> RunResult:
    """Load ``brand_id``, run the graph, and return the finished result.

    Raises ``HTTPException`` when there is no result to return: unknown brand, a gateway
    or checkpoint failure, or a graph that ended without a result. A run that finishes
    with status ``failed`` still returns its result.
    """
    try:
        brand = load_brand(brand_id)
    except BrandLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        with open_checkpointer(get_settings().checkpoint_db) as checkpointer:
            state = run_graph(brief, brand, checkpointer=checkpointer)
    except (GatewayError, OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    result = state["result"]
    if result is None:  # the graph always ends at the assembler, so this is a guard
        raise HTTPException(status_code=500, detail="the run finished without a result")
    return result


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure logs on startup, including when the app is served without the CLI."""
    configure_logging(get_settings())
    yield


app = FastAPI(
    title="BrandForge",
    version=__version__,
    summary="Turn a creative brief into on-brand ad copy.",
    description=(
        "POST /generate runs one brief through the same pipeline as `brandforge generate` "
        "and returns the run result, including the trace ID."
    ),
    lifespan=_lifespan,
)


@app.post(
    "/generate",
    tags=["generate"],
    summary="Generate on-brand ad copy for one brief.",
    response_description="The finished run, including trace_id (null when tracing is off).",
    responses={
        404: {"model": ErrorBody, "description": "No brand has this id."},
        500: {
            "model": ErrorBody,
            "description": "The run did not finish, so there is no result to return.",
        },
    },
)
def generate(
    body: Annotated[GenerateRequest, Body(openapi_examples={"voltride": _VOLTRIDE_EXAMPLE})],
) -> RunResult:
    """Run the brief through the pipeline and return the assembler's result.

    HTTP 200 means the pipeline finished. ``status`` in the body is ``complete``,
    ``partial`` or ``failed``. ``trace_id`` is the Langfuse trace, or null when tracing
    is off. A finished run with status ``failed`` is still HTTP 200.
    """
    return run_generation(body.brand, body.brief)
