# Promptfoo

Promptfoo runs the fixed eval set through two systems and puts their results side by side: the single-prompt baseline, and the full pipeline. The question the suite answers is whether the pipeline writes more on-brand copy than one prompt, and at what cost (ADR 0027).

Milestone M6 has the dataset (BF-33) and these providers (BF-34). Assertions, the judge, calibration, and the CI smoke test are still ahead. A run you start today returns copy, tokens, cost, and latency. It does not score that copy.

## What you can do now

These commands are safe to run. They load the case files and the providers. They do not call a model.

```bash
npx promptfoo@latest validate config --config evals/promptfooconfig.yaml
uv run pytest tests/test_eval_cases.py tests/test_promptfoo_providers.py
```

`validate config` checks the yaml and the test list. The pytest file checks the 30 cases and the provider wiring with the model calls mocked.

A real comparison is also available. It calls the model, so start with one case. The steps are under [Run one case](#run-one-case). From the promptfoo table you can read each system's variants, token counts, cost, and latency. With tracing on, a pipeline row includes a Langfuse trace id.

## Still to build in this milestone

| Task | What it will add | What you have instead |
| --- | --- | --- |
| BF-35 Deterministic assertions | Valid JSON, headline length, banned words, a call to action | The row is JSON, and the case file already records the caps and required phrases. Nothing checks them. |
| BF-36 Judge rubric | `llm-rubric` scores for brand voice, clarity, and CTA strength, on the judge tier | `BRANDFORGE_MODELS__JUDGE` is in config. This eval does not call it. |
| BF-37 Judge calibration | Your scores for 15 outputs in `evals/calibration/`, and an agreement script | That directory is not in the repo yet. |
| BF-38 Results and CI smoke | A committed summary in `evals/results/`, and a 5-case smoke eval on every push | CI runs Ruff, mypy, and pytest. The full eval is manual. |

Until BF-35 lands, promptfoo treats a row as a pass when the provider returns output. A green exit means both systems returned JSON. It does not mean the copy is on brand. An error row (a missing key, a gateway failure) fails the eval. Promptfoo then exits with code 100.

`flagged` is null on every baseline row, because the baseline has no critic. A flagged-rate comparison has to wait until a later check treats null as "not scored".

## How a run is wired

Promptfoo is a Node program. BrandForge stays in Python. The yaml in `evals/` is the only joint.

```mermaid
flowchart TD
  yaml["evals/promptfooconfig.yaml"]
  tests["evals/providers/tests.py"]
  cases["evals/cases/*.yaml"]
  basePy["evals/providers/baseline.py"]
  pipePy["evals/providers/pipeline.py"]
  pkg["brandforge.evals.promptfoo"]
  brands["src/brandforge/brands/*.yaml"]
  baseline["generate_baseline"]
  graph["run_graph"]
  prompts["src/brandforge/prompts/"]
  gateway["LLM gateway"]
  row["EvalOutput JSON plus tokenUsage and cost"]

  yaml --> tests
  yaml --> basePy
  yaml --> pipePy
  tests --> cases
  basePy --> pkg
  pipePy --> pkg
  pkg --> cases
  pkg --> brands
  pkg --> baseline
  pkg --> graph
  baseline --> prompts
  graph --> prompts
  baseline --> gateway
  graph --> gateway
  baseline --> row
  graph --> row
```

For one case, the call order is:

1. Promptfoo reads `evals/promptfooconfig.yaml`. Paths in that file are relative to `evals/`.
2. It loads `evals/providers/tests.py` and calls `generate_tests`. That function calls `load_cases`, which reads every YAML file in `evals/cases/`. One promptfoo test comes back per file, in filename order. The test description and `vars.case_id` are the case id. `metadata.brand_id` is the brand.
3. The prompt template is `{{case_id}}`. Promptfoo renders it to the case id. That string is the promptfoo prompt. It is not the text the model sees.
4. Promptfoo calls each provider's `call_api(prompt, options, context)` in a Python worker. The worker stays up between cases. `options` holds provider config such as the pipeline timeout. The provider functions do not read it. `context["vars"]["case_id"]` is the case to run. If that var is missing, the rendered prompt is the fallback.
5. `evals/providers/baseline.py` and `evals/providers/pipeline.py` forward the call to `brandforge.evals.promptfoo`. The scripts exist so promptfoo can load a file next to the config. The behaviour, and the tests, live in the package.
6. The package loads the case and the brand profile. The baseline calls `generate_baseline`. The pipeline calls `run_graph` with no checkpointer.
7. Those two functions build the model prompt from the versioned files in `src/brandforge/prompts/` and call the gateway. The gateway resolves the tier, validates the schema, retries, and counts tokens. The same code path serves `brandforge generate`.
8. The provider returns one JSON string in `output`, plus `tokenUsage` and `cost`, which promptfoo shows in its own columns.

The two providers are separate files, so promptfoo keeps their rows apart. The config also turns promptfoo's disk cache off. The cache key is the promptfoo prompt, which is only the case id, so a change to `prompts/baseline_v2.md` would not expire a cached row.

## The two systems

Both read the same brief and the same brand profile. They differ in how much work they do before the row comes back.

| | Baseline | Pipeline |
| --- | --- | --- |
| Entry | `generate_baseline` | `run_graph` |
| Model prompt | `prompts/baseline_<version>.md` | planner, writer, critic, and reviser prompts |
| Tier | `fast`, the same tier as the writer | `strong` for the planner and the critic, `fast` for the writer and the reviser |
| Critic and revisions | None | Up to `BRANDFORGE_BUDGETS__MAX_REVISIONS` (default 2) |
| Retrieval | None | On when `BRANDFORGE_RETRIEVAL_ENABLED` is true and an index exists |
| Status | `complete` when the call returns | `complete`, `partial`, or `failed`, copied from the run |
| `flagged` | null | The assembler's flag on each variant |
| `run_id`, `trace_id` | null | Copied from the run. `trace_id` is null when tracing is off |
| `numRequests` | 1 | Omitted. Repeat visits of a node are summed, so a count taken from that list would be short |
| Checkpoint | None | None. `brandforge inspect` cannot open an eval run |
| Worker timeout | Promptfoo's default, five minutes | 600000 ms (ten minutes), set on the pipeline provider |

A pipeline run still stops itself at the run budget. The defaults in `.env.example` are 60,000 tokens and 90 seconds, whichever comes first. The row then comes back `partial` or `failed` with whatever variants exist. That is inside the ten-minute worker limit. The worker limit is there so a stuck call does not sit forever.

Tracing follows the same settings as the CLI. With `BRANDFORGE_TRACING_ENABLED=true` and both Langfuse keys set, the pipeline row's `trace_id` opens the trace. The steps are in [langfuse.md](langfuse.md). The baseline is one gateway call outside a graph run, so it is not traced.

The eval provider does not call `configure_logging`. The JSON lines `brandforge generate` writes to stderr are a CLI behaviour. During an eval, promptfoo prints the progress. `npx promptfoo logs` shows promptfoo's own log.

Retrieval uses the index from `uv run brandforge index`. With no index, the pipeline logs a warning inside the process and writes the copy with no examples. `BRANDFORGE_RETRIEVAL_ENABLED=false` skips the search on purpose and still runs the rest of the graph, which is how you compare the pipeline with and without examples.

## The JSON row

`output` is one JSON object, the same shape for both systems. `EvalOutput` in `src/brandforge/evals/promptfoo.py` is that shape. The copy and the token numbers below illustrate the fields. The version strings are the ones on the Brightleaf profile.

```json
{
  "system": "pipeline",
  "case_id": "brightleaf_01_spring_blossom",
  "brand_id": "brightleaf",
  "brand_version": "1.0",
  "rubric_version": "1.0",
  "status": "complete",
  "run_id": "abc123",
  "trace_id": null,
  "revision_count": 1,
  "variants": [
    {
      "id": "v1",
      "channel": "email",
      "headline": "Spring tea, on the table",
      "body": "A limited blend for a quiet morning.",
      "cta": "Shop the blend",
      "flagged": false
    }
  ],
  "errors": [],
  "usage": {
    "input_tokens": 800,
    "output_tokens": 120,
    "cache_write_tokens": 0,
    "cache_read_tokens": 400,
    "total_tokens": 1320,
    "cost_usd": 0.012
  }
}
```

On a baseline row, `system` is `baseline`, `flagged` is null on every variant, and `run_id` and `trace_id` are null. `revision_count` is 0. `errors` is empty unless the pipeline recorded some. Each pipeline error is `node: message`.

`tokenUsage` is a second object, for promptfoo's columns:

| promptfoo field | Source |
| --- | --- |
| `tokenUsage.prompt` | `input_tokens` + `cache_write_tokens` + `cache_read_tokens` |
| `tokenUsage.completion` | `output_tokens` |
| `tokenUsage.total` | `usage.total_tokens`, which is the sum of all four counts |
| `tokenUsage.numRequests` | 1 on the baseline. Absent on the pipeline |
| `cost` | `usage.cost_usd` |

Latency in the promptfoo table is the wall clock of `call_api`: one gateway call for the baseline, and the whole graph for the pipeline.

When a case cannot run, the provider returns `{"error": "ErrorType: message"}` and no `output`. One bad case leaves the other cases, and the other provider, running. A programming bug (`AssertionError`) is the exception: that one propagates.

Headline caps and `must_mention` phrases live on the case as `hard_constraints`. The loader checks that they match the brief text. The provider passes the brief through, so the model is asked for them. Nothing in this milestone checks the reply against them.

## Before you run a model

From the repository root:

1. Install BrandForge and Node. Promptfoo is not a Python dependency. `npx` downloads it.

   ```bash
   uv sync
   node --version
   ```

2. Copy `.env.example` to `.env` and set the key for the provider your tiers use. The defaults are Anthropic models, so that key is `ANTHROPIC_API_KEY`. A `gemini:` tier needs `GEMINI_API_KEY`.

3. Point promptfoo at the project interpreter. That is the one that can import `brandforge`. The path is not pinned in the yaml, because Windows and POSIX virtualenvs differ.

   ```bash
   uv run python -c "import sys; print(sys.executable)"
   ```

   Bash:

   ```bash
   export PROMPTFOO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"
   ```

   PowerShell:

   ```powershell
   $env:PROMPTFOO_PYTHON = (uv run python -c "import sys; print(sys.executable)").Trim()
   ```

   A shell that already has that interpreter first on `PATH` can skip the variable. Promptfoo looks for `python` on `PATH` when the variable is unset.

4. Run from the repository root. Promptfoo loads `.env` from the current directory and the Python worker inherits those variables. BrandForge also reads `.env` from the worker's current directory. Passing `--env-file .env` makes the keys explicit.

5. Build the example index if you want the pipeline to retrieve approved copy. The baseline never retrieves.

   ```bash
   uv run brandforge index
   ```

The first index build downloads the embedding model. Later builds reuse it.

## Run one case

This runs `brightleaf_01_spring_blossom` through the baseline and the pipeline. That is two model-backed rows, and the pipeline row is several calls.

```bash
npx promptfoo@latest eval \
  --env-file .env \
  --config evals/promptfooconfig.yaml \
  --filter-pattern '^brightleaf_01_spring_blossom$'
```

The tests are in filename order, so this is the same single case:

```bash
npx promptfoo@latest eval \
  --env-file .env \
  --config evals/promptfooconfig.yaml \
  --filter-first-n 1
```

Open the latest run in a browser:

```bash
npx promptfoo@latest view
```

The output cell is the JSON row. The token and cost columns come from `tokenUsage` and `cost`. There is no assertion column that judges the copy.

Write a local copy of promptfoo's export if you want one. Pick a path outside `evals/results/`. That directory is reserved for the summary BF-38 will commit.

```bash
npx promptfoo@latest eval \
  --env-file .env \
  --config evals/promptfooconfig.yaml \
  --filter-first-n 1 \
  --output /tmp/brandforge-one-case.json
```

## Run a slice, or one system

`--filter-pattern` matches the test description, which is the case id. `--filter-metadata` matches `metadata.brand_id`. `--filter-providers` matches the provider id or the label (`baseline`, `pipeline`).

```bash
# Ten Brightleaf cases, both systems.
npx promptfoo@latest eval --env-file .env --config evals/promptfooconfig.yaml \
  --filter-pattern '^brightleaf_'

# The same ten, pipeline only.
npx promptfoo@latest eval --env-file .env --config evals/promptfooconfig.yaml \
  --filter-pattern '^brightleaf_' --filter-providers pipeline

# One brand, either spelling of the filter.
npx promptfoo@latest eval --env-file .env --config evals/promptfooconfig.yaml \
  --filter-metadata brand_id=ledgerly
```

The generator also accepts a `case_ids` list and keeps that order. The committed config does not set it, so the file runs all 30. To pin a list in the yaml, replace the `tests:` entry with:

```yaml
tests:
  - path: file://providers/tests.py:generate_tests
    config:
      case_ids:
        - brightleaf_01_spring_blossom
        - ledgerly_01_vat_reminders
```

An unknown id fails the run at startup with the known ids listed. Leave this change uncommitted if you only wanted a local slice. The CLI filters above do the same job without editing the file.

The full set is 30 cases, each through both systems. The config sets `maxConcurrency` to 1 so only one provider call runs at a time. Raise it for a long run with `--max-concurrency 2`. The pipeline's own model calls are still sequential inside that one call.

```bash
npx promptfoo@latest eval --env-file .env --config evals/promptfooconfig.yaml
```

A failed row can be retried without repeating the successes:

```bash
npx promptfoo@latest eval --env-file .env --config evals/promptfooconfig.yaml --retry-errors
```

## Compare retrieval on and off

Run the same case twice. The cache is off, so the second run calls the model again. The baseline rows should match in shape. The pipeline rows differ in whether approved examples were in the writer prompt.

```bash
npx promptfoo@latest eval --env-file .env --config evals/promptfooconfig.yaml \
  --filter-first-n 1 --output /tmp/brandforge-retrieval-on.json

BRANDFORGE_RETRIEVAL_ENABLED=false npx promptfoo@latest eval \
  --env-file .env --config evals/promptfooconfig.yaml \
  --filter-first-n 1 --output /tmp/brandforge-retrieval-off.json
```

PowerShell:

```powershell
$env:BRANDFORGE_RETRIEVAL_ENABLED = "false"
npx promptfoo@latest eval --env-file .env --config evals/promptfooconfig.yaml `
  --filter-first-n 1 --output $env:TEMP\brandforge-retrieval-off.json
```

A shell variable wins over the same name in `.env`. Set it back, or open a new shell, before the next run.

## Call one provider from Python

The same functions promptfoo calls are importable. This still uses the model. It is a way to inspect one row without the Node runner.

```bash
uv run python -c "import json; from brandforge.evals import call_baseline; print(json.dumps(call_baseline('brightleaf_01_spring_blossom', {}, {}), indent=2))"
```

`call_pipeline` is the graph. `generate_tests()` is the test list, and it does not call a model.

## Files

| Path | Role |
| --- | --- |
| `evals/promptfooconfig.yaml` | Providers, the case-id prompt, the test generator, cache off, concurrency 1 |
| `evals/providers/baseline.py` | Promptfoo `call_api` for the baseline |
| `evals/providers/pipeline.py` | Promptfoo `call_api` for the pipeline |
| `evals/providers/tests.py` | Promptfoo `generate_tests` |
| `evals/cases/*.yaml` | The 30 cases (ADR 0026) |
| `src/brandforge/evals/promptfoo.py` | The row shape and the two provider functions |
| `src/brandforge/evals/cases.py` | Case loader |
| `tests/test_promptfoo_providers.py` | Provider tests with the model mocked |
| `docs/adr/0027_promptfoo_providers.md` | Why the prompt is a case id and why both systems share one row |
