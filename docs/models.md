## **Data contracts (`models.py`)**

Every piece of data that crosses an agent boundary in BrandForge is a Pydantic model. Agents are pure functions over shared state, so these models are the only way they communicate. They are strict by design: unknown fields are rejected, strings are stripped and must be non-empty, and numeric ranges are enforced. A malformed LLM output fails at the boundary and becomes a retryable error through the gateway's schema-repair step, rather than a crash further down the graph.

### **Models and where they sit in the pipeline**

| Model | Purpose | Produced by | Consumed by |
| :---- | :---- | :---- | :---- |
| **Brief** | The user's request: product, audience, objective, channels, constraints. Channels must be non-empty and unique. | CLI, API or eval case | Planner, writer |
| **BrandProfile** | A brand's voice, do/don't lists, banned words and rubric, with a version recorded in each result for reproducible evals. | Brand store (YAML loader) | Planner, writer, critic |
| **Rubric / RubricCriterion** | The scoring criteria for a brand. Each criterion must define anchors for all levels 1 to 5, so the judge scores against concrete descriptions. | Part of BrandProfile | Critic, eval judge |
| **Variant** | One piece of ad copy: channel, headline, body and call to action. | Writer, reviser | Critic, assembler |
| **Critique** | The critic's verdict on one variant: per-criterion scores (1 to 5), overall score, pass/fail and fix notes. | Critic | Router, reviser, assembler |
| **Usage** | Token counts and cost. Supports `+` so it can accumulate across nodes. | LLM gateway | Run state, assembler, cost reporting |
| **Plan** | The planner's reading of a brief: audience, angle, channels and variants per channel. | Planner (BF-13) | Writer |
| **Example** | An approved piece of copy for a brand, retrieved as a style reference. | Retriever (BF-31) | Writer |
| **RunError** | A failure recorded in state (node and message), so a run can still end in a defined status. | Any node (BF-22) | Assembler |
| **RunState** | The `TypedDict` that is the single source of truth for a run. `usage` and `errors` carry reducers (`add_usage`, `add_errors`), so a node returns only its own usage or new errors and the graph accumulates them. `new_run_state` builds the initial state. | Graph entry point | Every node |

### **How they flow**

1. A **Brief** and a **BrandProfile** enter the run and form the input half of the state.  
2. The writer turns them into **Variants**, validated against the schema as they come back from the model.  
3. The critic scores each variant against the brand's **Rubric** and returns **Critiques**.  
4. The router (plain code) reads the critiques: if every variant passes, it assembles; if some fail, it routes them to the reviser, capped at two passes.  
5. The reviser rewrites only failing variants using each critique's `fixes`.  
6. Every gateway call returns a **Usage**. These are summed into the run's total, which enforces the token budget and is reported with the result.

### **Why this design**

* **Quality is measurable.** Anchored rubrics and bounded scores make critic and judge outputs comparable across runs.  
* **Reliability.** Validation at every boundary means bad output is caught where it is produced, which supports the "every run ends in a defined state" goal.  
* **Modifiability.** Agents depend on these contracts, not on each other, so one can be swapped without touching the rest.  
* **Reproducibility.** The profile version travels with the result, so an eval run can be tied to the exact brand configuration it used.

### **The graph shell (`graph.py`)**

`build_graph()` compiles a LangGraph `StateGraph` over `RunState`, and `run_graph(brief, brand)` runs one brief and returns the final state. The graph currently runs the `planner` node (BF-13), which writes `plan`, and then a `baseline` node that wraps `generate_baseline`, so `brandforge generate` keeps working end to end through the graph. The baseline does not use the plan yet; the writer (BF-14) replaces it. Errors from a node still propagate; BF-22 turns them into recorded `RunError`s and a `partial` or `failed` status.
