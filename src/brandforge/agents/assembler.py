"""Assembler: packages the final result and owns the run's final status (BF-18).

Plain code, no model call. Every path through the graph ends here, whether the router said
`assemble` (every variant passed) or `stop` (some did not and the run must not try again), so
a run always produces a `RunResult` with an explicit status.

Flagging reuses the router's `failing_variant_ids`: a variant is flagged when it has no passing
critique, which includes one that was never scored. The router and the assembler therefore
cannot disagree about what failed.
"""

from typing import Any

from brandforge.models import FinalStatus, RunResult, RunState, VariantResult
from brandforge.router import failing_variant_ids


def _final_status(*, variant_count: int, flagged_count: int, error_count: int) -> FinalStatus:
    """No variants is `failed`. Variants plus any error or flag is `partial`. Otherwise
    `complete`."""
    if variant_count == 0:
        return "failed"
    if error_count or flagged_count:
        return "partial"
    return "complete"


def assemble_result(state: RunState) -> dict[str, Any]:
    """Graph node: reads the full state; returns `result` and the final `status`.

    Variants keep their order. Each carries the latest critique for its id, or `None` if it
    has none. Errors recorded by earlier nodes are copied into the result unchanged.
    """
    flagged = set(failing_variant_ids(state))
    critiques = {critique.variant_id: critique for critique in state["critiques"]}
    items = [
        VariantResult(
            variant=variant,
            critique=critiques.get(variant.id),
            flagged=variant.id in flagged,
        )
        for variant in state["variants"]
    ]
    status = _final_status(
        variant_count=len(items),
        flagged_count=len(flagged),
        error_count=len(state["errors"]),
    )
    brand = state["brand"]
    result = RunResult(
        run_id=state["run_id"],
        status=status,
        brand_id=brand.id,
        brand_version=brand.version,
        rubric_version=brand.rubric.version,
        variants=items,
        revision_count=state["revision_count"],
        errors=list(state["errors"]),
        usage=state["usage"],
    )
    return {"result": result, "status": status}
