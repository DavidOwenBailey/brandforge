# **BrandForge Architecture Design**

Oct 4, 2026 · David Bailey

BrandForge is a Python multi-agent pipeline, orchestrated with LangGraph, that turns a creative brief into scored, on-brand ad copy variants. It is built as a one-week proof of concept, but designed so that the path to production is clear.

## **Goals, scope and quality attributes**

**Goals**

1. Generate ad copy variants that measurably fit a brand's voice better than a single-prompt baseline.  
2. Show production-grade agent engineering: typed state, bounded loops, graceful failure, full tracing.  
3. Make quality measurable and repeatable through an automated evaluation suite.

**In scope:** text ad copy for up to four channels (search, social, display, email subject), CLI and minimal web UI, three fictional brands, a local vector store, offline evaluation.

**Out of scope (next steps):** image or video generation, fine-tuning, multi-tenant auth, cloud deployment, human-review UI.

**Quality attributes, in priority order**

| Attribute | Target for the POC | How it's achieved |
| :---- | :---- | :---- |
| Output quality | Mean brand-voice score above baseline on the eval set | Critic loop, retrieved brand examples, calibrated judge |
| Reliability | Every run ends in a defined state, never a crash or infinite loop | Schema validation, bounded retries, revision cap |
| Observability | Every agent step traceable with inputs, outputs, tokens, latency | Langfuse callbacks on every node |
| Cost | Known cost per run, reported with each result | Token accounting in state; cheaper model for drafting |
| Latency | One brief end to end in under \~60 seconds | Parallel variant critique; small per-agent contexts |
| Modifiability | Swap model provider or add an agent without touching others | Provider abstraction; agents as pure functions over state |

The latency target is a working assumption to confirm on day 3\.

## **System overview**

![BrandForge system overview · interfaces, orchestrator, shared services](img/system_overview.png)

Interfaces, orchestrator, shared services Interfaces and the eval harness invoke the same compiled graph. Agents never call each other or a provider directly: the router owns control flow, and every model call goes through the gateway, which handles retries, token accounting and tracing.

## **Components**

Each agent is a pure function: it reads the slice of state it needs and returns a partial state update. No agent calls another directly; the graph decides what runs next.

| Component | Responsibility | Reads | Writes | Model tier |
| :---- | :---- | :---- | :---- | :---- |
| Planner | Turn the brief into a structured plan: audience, angle, channels, variant count | brief, brand | plan | Strong |
| Retriever | Fetch the 3–5 most relevant approved examples for this brand and channel | brand.id, plan | examples | Embeddings only |
| Writer | Produce variants per channel as schema-valid JSON | plan, brand, examples | variants | Fast |
| Brand critic | Score each variant 1–5 on voice, clarity, call to action; explain fixes | variants, brand.rubric | critiques | Strong |
| Router | Decide: all pass, revise, or stop and flag | critiques, revision\_count | next node | None (code) |
| Reviser | Rewrite only failing variants using the critic's notes | failing variants, critiques | variants, revision\_count | Fast |
| Assembler | Package final variants, scores, flags, cost and trace ID | full state | result | None (code) |
| LLM gateway | One interface over Anthropic and OpenAI SDKs; retries, timeouts, token accounting | prompts | completions, usage | — |
| Brand store | Brand profiles and rubrics as versioned YAML files | — | — | — |
| Example index | Embedded approved copy per brand (Chroma, local) | — | — | — |
| Interfaces | Typer CLI and a thin FastAPI endpoint with a Streamlit page | brief input | result | — |

"Strong" and "fast" are tiers in config, not hard-coded models, so the cost/quality trade-off can be tuned per agent.

## **State model and data contracts**

All data crossing an agent boundary is a Pydantic model. The graph state is the single source of truth for a run, so any run can be replayed from a checkpoint.

```python
class Brief(BaseModel):
    product: str
    audience: str
    objective: Literal["awareness", "consideration", "conversion"]
    channels: list[Channel]  # search | social | display | email
    constraints: list[str] = []  # e.g. "no discounts", "max 90 chars"


class BrandProfile(BaseModel):
    id: str
    voice: list[str]  # "warm", "plain-spoken", ...
    do: list[str]
    dont: list[str]
    banned_words: list[str]
    rubric: Rubric  # criteria + 1-5 anchors


class Variant(BaseModel):
    id: str
    channel: Channel
    headline: str
    body: str
    cta: str


class Critique(BaseModel):
    variant_id: str
    scores: dict[str, int]  # criterion -> 1..5
    overall: float
    passed: bool
    fixes: list[str]


class RunState(TypedDict):
    run_id: str
    brief: Brief
    brand: BrandProfile
    plan: Plan | None
    examples: list[Example]
    variants: list[Variant]
    critiques: list[Critique]
    revision_count: int
    errors: list[RunError]
    usage: Usage  # tokens + cost, accumulated per node
    status: Literal["running", "complete", "partial", "failed"]
```  
**Contract rules**

* Every LLM output is parsed into its model; a parse failure is a retryable error, not a crash.  
* Agents return only the keys they own (see Components). usage and errors use reducer functions so nodes append rather than overwrite.  
* Brand profiles and rubrics carry a version; the result records which version was used, so eval runs are reproducible.

## **Control flow and failure handling**

The graph is planner → retriever → writer → critic → router, where the router sends failing variants to the reviser and back to the critic at most twice. Every path ends at the assembler, so a run always produces a result with an explicit status.

**Failure handling, by layer**

| Failure | Where it's caught | Handling | Run status |
| :---- | :---- | :---- | :---- |
| API timeout, rate limit, 5xx | LLM gateway | Exponential backoff with jitter, 3 attempts (tenacity) | Continues if a retry succeeds |
| Output fails schema validation | LLM gateway | Re-ask once with the validation error in the prompt | Continues, or node fails |
| Node fails after retries | Graph error edge | Record in errors, route to assembler with what exists | partial or failed |
| Variant still below threshold after 2 revisions | Router | Stop revising; mark variant flagged | partial |
| Retriever returns nothing | Retriever | Proceed with zero examples; log a warning | Continues |
| Run exceeds token budget | Gateway budget check | Stop further revisions; assemble current best | partial |

**Design principles**

* **Bounded everything:** retries, revisions, tokens and wall-clock time all have hard limits in config.  
* **Partial over nothing:** a run that has some passing variants returns them, clearly labelled, rather than failing.  
* **Revise only what failed:** the reviser touches failing variants only, which saves tokens and prevents regressions in good ones.  
* **Checkpointing:** LangGraph's SQLite checkpointer stores state after each node, so a failed run can be inspected or resumed.

This is the same compensation thinking as a Saga: each step has a defined outcome on failure, and the orchestrator, not the step, decides what happens next.

## **Proposed tech stack**

The stack favours mainstream, well-documented tools that agent roles ask for by name, and runs entirely on a laptop with one API key.

| Layer | Choice | Why | Alternative considered |
| :---- | :---- | :---- | :---- |
| Language | Python 3.12 | Lingua franca of agent tooling; supported by LangGraph 1.2.x (Python 3.10–3.13) | TypeScript (LangGraph.js) |
| Project tooling | uv, Ruff, mypy, pytest | Fast, reproducible setup; strict typing feels familiar coming from C\# | Poetry, Black |
| Orchestration | LangGraph 1.2.x | Explicit graph and typed state, conditional edges, checkpointing; closest to BPMN-style orchestration | CrewAI (less control over state), AutoGen (conversation-centric) |
| Data contracts | Pydantic v2 | Validates every LLM output; generates JSON schemas for structured output | dataclasses \+ jsonschema |
| LLM provider | Anthropic SDK, behind an internal gateway | Tool use and structured outputs; gateway keeps the provider swappable | OpenAI SDK, LiteLLM |
| Strong tier | Claude Sonnet 5.5 (claude-sonnet-5-5, \$2 / \$10 per M tokens) | Planner and critic need judgement; best speed/quality balance | Claude Opus 5.5 |
| Fast tier | Claude Haiku 4.5 (claude-haiku-4-5-20251001, \$1 / \$5 per M tokens) | Writer and reviser run most often; cheapest and fastest | Sonnet for both tiers |
| Judge model | Claude Opus 5.5, optionally cross-checked with an OpenAI model | A stronger, different model than the generator reduces self-preference bias | Same model as critic (cheaper, more biased) |
| Retries | tenacity | Declarative backoff and retry policies | Hand-rolled loops |
| Retrieval | Chroma (local, persistent) \+ provider embeddings | Zero-infrastructure vector store; enough for a few hundred examples | pgvector, Qdrant |
| Brand store | Versioned YAML files in the repo | Reviewable in Git; no database needed for the POC | Postgres |
| Checkpointing | LangGraph SQLite checkpointer | Inspect and resume runs locally | Postgres checkpointer (production) |
| Tracing | Langfuse (self-hosted via Docker Compose, or free cloud tier) | Open source; native LangGraph integration; shows agent graphs, tokens, cost | LangSmith, OpenTelemetry \+ Jaeger |
| Evaluation | promptfoo \+ LLM-as-a-judge rubric; pytest for deterministic checks | Named in target job specs; YAML test suites run in CI | DeepEval, Ragas |
| Prompt optimisation (stretch) | DSPy | Systematic prompt tuning against the eval set | Manual iteration |
| Interfaces | Typer CLI; FastAPI endpoint; Streamlit demo page | CLI for evals and scripting, API to show service design, Streamlit for the video | Gradio |
| Packaging and CI | Docker, GitHub Actions | One-command run; lint, tests and a small eval subset on every push | — |

Model IDs and prices are from Anthropic's [models overview](https://platform.claude.com/docs/en/models/overview); LangGraph version from [PyPI](https://pypi.org/project/langgraph/). Pin exact versions in pyproject.toml on day one.

## **Evaluation architecture**

The evaluation answers one question: does the full pipeline produce better on-brand copy than a single prompt, and at what cost? It runs offline against a fixed dataset, so results are comparable between changes.

**Dataset:** 3 brands × 10 briefs \= 30 cases, stored as YAML under evals/. Each case has the brief, the brand ID, and any hard constraints.

**Two systems under test**

1. **Baseline:** one prompt with the brief and brand profile, same model as the writer.  
2. **Pipeline:** the full graph.

**Three layers of checks**

| Layer | Tool | What it checks | Pass rule |
| :---- | :---- | :---- | :---- |
| Deterministic | promptfoo assertions \+ pytest | Valid JSON, character limits per channel, banned words absent, required CTA present | 100% must pass |
| Model-graded | promptfoo llm-rubric with the judge model | Brand-voice fit, clarity, CTA strength, each scored 1–5 against anchored descriptions | Report mean and distribution |
| Human calibration | Your own scores on a 15-case sample | Agreement between you and the judge | Report agreement; tune rubric if it's low |

**Reported per run:** mean score per criterion, pass rate, flagged-variant rate, tokens, cost and latency per brief, all for baseline vs. pipeline side by side.

**Judge bias controls**

* Judge is a different, stronger model than the generator.  
* Rubric anchors describe each score level concretely, with an example.  
* Variants are scored individually, not compared side by side, to avoid position bias.  
* Length limits are enforced deterministically, so the judge isn't rewarding verbosity.

**In CI:** lint, unit tests and a 5-case eval smoke test on every push; the full 30-case run is manual, with results committed to evals/results/.

## **Observability, cost and security**

**Observability**

* One Langfuse trace per run, with a span per graph node; each span records model, prompt version, input, output, tokens, latency and errors.  
* The trace ID is returned in the result and printed by the CLI, so any output links straight to how it was produced.  
* Structured JSON logs (structlog) carry the same run\_id for correlation.  
* Prompts are versioned files; the prompt version is a span attribute, so eval results tie to exact prompts.

**Cost controls**

* The gateway accumulates tokens and cost into state after every call.  
* Per-run token budget in config; when hit, revisions stop and the best current result is assembled.  
* Fast-tier model for the high-volume writer and reviser calls; strong tier only for planning, critique and judging.  
* Prompt caching for the static parts of prompts (system prompt, brand profile), which repeat on every call in a run.  
* A spending limit on the API account during development.

**Security and data handling**

* API keys from environment variables via pydantic-settings; .env git-ignored; nothing secret in traces.  
* Brief text is treated as untrusted input: it's placed in a clearly delimited section of the prompt, and outputs are schema-validated, so injected instructions can't change the output shape.  
* No real brand or customer data in the repo; all three brands are fictional.  
* The FastAPI endpoint is local-only for the POC; production would add auth, rate limiting and tenant isolation per brand.

## **Repository structure**
```text
brandforge/  
├── README.md                 # problem, diagram, quickstart, results, limitations  
├── pyproject.toml            # pinned deps, ruff, mypy, pytest config  
├── docker-compose.yml        # app \+ optional self-hosted Langfuse  
├── .env.example  
├── docs/  
│   ├── architecture.md       # this document  
│   └── adr/                  # 0001-langgraph.md, 0002-revision-cap.md, ...  
├── src/brandforge/  
│   ├── config.py             # settings, model tiers, budgets  
│   ├── models.py             # Pydantic contracts \+ RunState  
│   ├── graph.py              # nodes, edges, router, checkpointer  
│   ├── agents/               # planner.py, writer.py, critic.py, reviser.py, assembler.py  
│   ├── prompts/              # versioned prompt templates  
│   ├── llm/gateway.py        # provider abstraction, retries, token accounting  
│   ├── retrieval/            # Chroma index build \+ query  
│   ├── brands/               # brand profiles and rubrics (YAML)  
│   └── interfaces/           # cli.py (Typer), api.py (FastAPI), app.py (Streamlit)  
├── evals/  
│   ├── cases/                # 30 brief cases (YAML)  
│   ├── promptfooconfig.yaml  # baseline vs pipeline providers, assertions, rubric  
│   ├── calibration/          # your hand scores  
│   └── results/              # committed run summaries  
├── tests/                    # unit tests: router logic, schemas, gateway (LLM mocked)  
└── .github/workflows/ci.yml  # lint, type-check, tests, 5-case eval smoke test
```
## **Decisions and open questions**

| ADR | Decision | Reason | Trade-off accepted |
| :---- | :---- | :---- | :---- |
| 0001 | LangGraph for orchestration | Explicit state machine with checkpointing; control over every transition | More code than CrewAI's role-based setup |
| 0002 | Workflow with one evaluator loop, not autonomous agents | The task has a known shape; a fixed graph is cheaper, testable and predictable | Less flexibility for unusual briefs |
| 0003 | Revision loop capped at 2 passes; flag rather than fail | Bounded cost and latency; partial results are still useful | Some variants ship flagged instead of fixed |
| 0004 | Two model tiers set in config | Cuts cost on high-volume calls; tiers can be tuned per agent from eval data | Fast tier may need more revisions |
| 0005 | Judge model differs from generator | Reduces self-preference bias in scores | Higher eval cost |
| 0006 | Pydantic contracts at every boundary | Malformed output is caught at the source and retried | Slightly more prompt and schema work |
| 0007 | Local-first (Chroma, SQLite, YAML) | Clone-and-run in minutes; no infrastructure to explain | Not multi-user; production swaps to Postgres/pgvector |

**Open questions**

* What pass threshold per criterion gives the best quality/cost balance? Decide from eval data on day 5 (starting point: overall ≥ 4.0, no criterion below 3).  
* Should the critic score variants in parallel (faster) or in one call (cheaper, shared context)? Measure both on day 3\.  
* Is retrieval worth its complexity here? Compare eval scores with and without examples.

**Path to production (not built)**

Postgres checkpointer and pgvector, multi-tenant brand isolation, auth on the API, a human-review queue for flagged variants, eval gates on prompt changes in CI, and online A/B testing of variants against real performance data.