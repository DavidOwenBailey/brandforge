## **Data contracts (`models.py`)**

Every piece of data that crosses an agent boundary in BrandForge is a Pydantic model. Agents are pure functions over shared state, so these models are the only way they communicate. They are strict by design: unknown fields are rejected, strings are stripped and must be non-empty, and numeric ranges are enforced. A malformed LLM output fails at the boundary and becomes a retryable error through the gateway's schema-repair step, rather than a crash further down the graph.

### **Models and where they sit in the pipeline**

| Model | Purpose | Produced by | Consumed by |
| :---- | :---- | :---- | :---- |
| **Brief** | The user's request: product, audience, objective, channels, constraints. Channels must be non-empty and unique. | CLI, API or eval case | Planner, writer |
| **BrandProfile** | A brand's voice, do/don't lists, banned words and rubric, with a version recorded in each result for reproducible evals. | Brand store (YAML loader) | Planner, writer, critic |
| **Rubric / RubricCriterion** | The scoring criteria for a brand. Each criterion must define anchors for all levels 1 to 5, so the judge scores against concrete descriptions. | Part of BrandProfile | Critic, eval judge |
| **Variant** | One piece of ad copy: channel, headline, body and call to action. | Writer, reviser | Critic, assembler |
| **Critique** | The critic's verdict on one variant: per-criterion scores (1 to 5), overall score, pass/fail and fix notes. Built in code by `build_critique` from a `CriticReply`, never returned by the model directly. | Critic (BF-15) | Router, reviser, assembler |
| **CriticReply** | What the critic model returns for one variant: a list of `{criterion, score}` items and fix notes. Lists, not a dict, so the schema is portable; `build_critique` checks every rubric criterion is scored once, that scores are 1 to 5, and computes `overall` and `passed` from the configured thresholds. Lives in `scoring.py`. | Critic model | `build_critique` |
| **Usage** | Token counts and cost. Supports `+` so it can accumulate across nodes. | LLM gateway | Run state, assembler, cost reporting |
| **Plan** | The planner's reading of a brief: audience, angle, channels and variants per channel. | Planner (BF-13) | Writer |
| **Example** | An approved piece of copy for a brand, retrieved as a style reference. | Retriever (BF-31) | Writer |
| **VariantResult** | One variant in the final result: the variant, its latest critique (or none if it was never scored) and whether it is flagged. | Assembler (BF-18) | CLI, API, evals |
| **RunResult** | What a finished run hands back: final status, brand and rubric versions, one `VariantResult` per variant, revision count, errors and total usage. `flagged_count` says how many variants are not shown to be on brand. | Assembler (BF-18) | CLI, API, evals |
| **RunError** | A failure recorded in state (node, message and `fatal`), so a run can still end in a defined status. `fatal` is set by the graph's error edges when a node raised and the run was sent to the assembler; a non-fatal error is a warning the run carried on past. | Writer (warnings), graph error guard (BF-22) | Assembler, CLI |
| **RunState** | The `TypedDict` that is the single source of truth for a run (`result` stays `None` until the assembler fills it; `started_at` is when the run began, in epoch seconds, and is what the wall-clock budget runs from). `usage` and `errors` carry reducers (`add_usage`, `add_errors`), so a node returns only its own usage or new errors and the graph accumulates them. `new_run_state` builds the initial state. | Graph entry point | Every node |

### **How they flow**

1. A **Brief** and a **BrandProfile** enter the run and form the input half of the state.  
2. The writer turns them into **Variants**, validated against the schema as they come back from the model.  
3. The critic scores each variant against the brand's **Rubric** and returns **Critiques**.  
4. The router (plain code) reads the critiques: if every variant passes, it assembles; if some fail, it routes them to the reviser, capped at two passes, and when the cap, the token budget or the wall-clock budget is reached it stops and assembles what exists.  
5. The reviser rewrites only failing variants using each critique's `fixes`.  
6. The assembler packages the variants, their critiques, flags, errors and usage into a **RunResult** and sets the final status.  
7. Every gateway call returns a **Usage**. These are summed into the run's total, which is reported with the result and counts against the token budget. Inside a node the gateway checks the run budget (tokens and time) before every model call and stops the node with `BudgetExceededError` once it is used up (BF-23, ADR 0017).

### **Why this design**

* **Quality is measurable.** Anchored rubrics and bounded scores make critic and judge outputs comparable across runs.  
* **Reliability.** Validation at every boundary means bad output is caught where it is produced, which supports the "every run ends in a defined state" goal.  
* **Modifiability.** Agents depend on these contracts, not on each other, so one can be swapped without touching the rest.  
* **Reproducibility.** The profile version travels with the result, so an eval run can be tied to the exact brand configuration it used.

### **The graph shell (`graph.py`)**

`build_graph()` compiles a LangGraph `StateGraph` over `RunState`, and `run_graph(brief, brand)` runs one brief and returns the final state. The graph currently runs the `planner` node (BF-13), which writes `plan`, then the `writer` node (BF-14), which writes `variants` from the plan, the brand profile and any `examples` in state (empty until the retriever exists), and then the `critic` node (BF-15), which writes one `Critique` per variant into `critiques`. After the critic, the router (BF-16) decides: `revise` runs the `reviser` node (BF-17), which rewrites only the failing variants from the critic's `fixes`, adds one to `revision_count` and sends the run back to the critic, at most twice by default; `assemble` and `stop` both go to the `assembler` node (BF-18), which is the only node that sets `status`: it stays `running` until then. The assembler flags every variant without a passing critique (using the router's `failing_variant_ids`) and writes a `RunResult` into `result`. The status is `failed` if there are no variants, `partial` if any variant is flagged or any `RunError` was recorded (for example a channel that came back short), and `complete` otherwise. The single-prompt baseline is not in the graph; the evals call `generate_baseline` directly. The planner, writer, critic and reviser each run inside a guard (BF-22): if one raises after the gateway's retries, the guard records a fatal `RunError` (and the usage of any failed calls), the edge after that node goes straight to the assembler, and the run ends `partial` or `failed` instead of raising. See ADR 0016.
