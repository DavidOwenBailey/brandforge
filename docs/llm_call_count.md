# LLM calls per graph node

For a `uv run brandforge generate --brand voltride --brief src/brandforge/briefs/voltride_01_commuter_ebike.yaml` run, the call count is fixed for four nodes and depends on the planner's variant count and the revision loop for the other two.

The brief asks for two channels, `search` and `social`. With the current settings (`PLANNER_MAX_VARIANTS_PER_CHANNEL=5`, `MAX_REVISIONS=2`), a run that finishes the loop looks like this:

| Node | Tier | LLM calls |
| :---- | :---- | :---- |
| Planner | strong | **1** |
| Retriever | none | **0** |
| Writer | fast | **2** |
| Critic | strong | **2N × (1 + R)** |
| Reviser | fast | **one per failing variant, on each revision pass** |
| Assembler | none | **0** |

`N` is the planner's `variants_per_channel`, clamped in code to 1–5. `R` is how many revision passes actually run: 0, 1, or 2.

## Planner

The planner makes one `complete_structured` call on the strong tier. Channels stay the brief's two channels. The model only chooses the audience, the angle, and `N`.

## Retriever

The retriever does not call a model. Retrieval is on, so it runs two local embedding searches (one per channel) against the Voltride Chroma index.

## Writer

The writer makes one fast-tier call per planned channel, so two calls. Each call is asked for `N` variants. Ids such as `search-1` are assigned in code.

## Critic

The critic makes one strong-tier call per variant. After the writer that is `2N` calls. If any variant fails and the run still has revision and token or wall-clock budget, the reviser runs and the critic then scores every variant again, including ones that already passed. That adds another `2N` calls per pass. With `MAX_REVISIONS=2`, the critic can run three times: `2N`, `4N`, or `6N` calls.

## Reviser

The reviser makes one fast-tier call per failing variant, and only for those variants. If the first critique passes everything, the reviser node is never entered, so it makes 0 calls. If every variant fails both passes, it makes `2N` calls per pass, `4N` in total.

## Assembler

The assembler only packages state. It makes no model call.

## Bounds

The smallest full run is `N=1` and a pass on the first critique: 1 + 2 + 2 = **5** calls. The largest, `N=5` and every variant failing both revision passes, is 1 + 2 + 30 + 20 = **53** calls.

## What one call counts as

Each of those counts is one gateway call. A reply that fails schema validation gets one repair request (`max_schema_repairs` is 1). A transient provider error is retried inside that same call, up to 3 attempts. A node that raises stops the run, so later calls in that node are not made.
