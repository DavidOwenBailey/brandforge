# Promptfoo

Promptfoo runs the fixed eval set through two systems and puts their results side by side: the single-prompt baseline, and the full pipeline. The question the suite answers is whether the pipeline writes more on-brand copy than one prompt, and at what cost (ADR 0027).

Milestone M6 has the dataset (BF-33), the providers (BF-34), the deterministic assertions (BF-35), and the judge rubric (BF-36). Calibration and the CI smoke test are still ahead. A run returns copy, tokens, cost, and latency. It scores that copy against the hard checks, and the judge scores each variant from 1 to 5 on that brand's anchors.

## What you can do now

These commands are safe to run. They load the case files and the providers. They do not call a model.

```bash
npx promptfoo@latest validate config --config evals/promptfooconfig.yaml
uv run pytest tests/test_eval_cases.py tests/test_promptfoo_providers.py tests/test_assertions.py tests/test_judge.py
```

`validate config` checks the yaml and the test list. The pytest files check the 30 cases, the provider wiring, the deterministic checks, and the judge grader. The model calls are mocked.

A real comparison is also available. It calls the model, so start with one case. The steps are under [Run one case](#run-one-case). From the promptfoo table you can read each system's variants, token counts, cost, latency, the deterministic checks, and the judge's 1–5 mean. With tracing on, a pipeline row includes a Langfuse trace id. The deterministic checks do not call a model. The judge rubric does: one gateway call per variant, on the judge tier.

## Still to build in this milestone

| Task | What it will add | What you have instead |
| --- | --- | --- |
| BF-37 Judge calibration | Your scores for 15 outputs in `evals/calibration/`, and an agreement script | That directory is not in the repo yet. |
| BF-38 Results and CI smoke | A committed summary in `evals/results/`, and a 5-case smoke eval on every push | CI runs Ruff, mypy, and pytest. The full eval is manual. |

The deterministic checks are valid JSON, the eval-row schema, headline caps, the requested channels, banned words, a call to action, and any required phrase. They must all pass. They do not grade brand voice. The judge rubric does that, and [its section](#the-judge-rubric) says how a finished grade is recorded. An error row (a missing key, a gateway failure) has no output, so it fails the eval. Promptfoo then exits with code 100.

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
  checks["deterministic assertions"]
  judge["judge rubric, one call per variant"]

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
  row --> checks
  row --> judge
  judge --> gateway
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
9. promptfoo runs the `defaultTest` assertions on that output. `is-json` checks the syntax. The Python checks load the case and the brand and score the row. They do not call a model. `evals/providers/assertions.py` forwards to `brandforge.evals.assertions`, the same split as the providers.
10. `generate_tests` also attaches one `llm-rubric` per case, using that brand's anchors. `evals/providers/judge.py` forwards to `brandforge.evals.judge`. The grader calls the gateway once per variant, on the judge tier, with `prompts/judge_v1.md`. The model sees one variant. The row score is the mean of the criterion means. When a cross-check model is configured, a second rubric grades the same anchors again. The details are in [The judge rubric](#the-judge-rubric).

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

Headline caps and `must_mention` phrases live on the case as `hard_constraints`. The loader checks that they match the brief text. The provider passes the brief through, so the model was asked for them. The checks below score the reply.

## Deterministic assertions

Every test runs the same checks. They are `defaultTest` in `evals/promptfooconfig.yaml`. pytest calls the same functions. Neither path calls a model (ADR 0028).

| Check | Metric | Passes when |
| --- | --- | --- |
| Valid JSON | `valid_json` | The output parses as JSON |
| Eval row | `eval_row` | That JSON is an `EvalOutput` for this case and this brand |
| Channel limits | `channel_limits` | Every requested channel has a variant, no extra channel appears, and each headline is within `headline_max_chars` when that channel has a cap |
| Banned words | `banned_words` | No brand banned word appears in a headline, body, or call to action |
| Call to action | `cta_present` | Every variant has a non-empty call to action, and the row has at least one variant |
| Required phrase | `must_mention` | Every variant includes each `must_mention` phrase. A case with none still needs variants |

A row fails when any check fails. The score is 1 or 0. `status` may be `partial` or `failed` and the JSON checks still pass. The copy checks then fail if the copy is missing or over a cap.

The cap is the case's `headline_max_chars`. The email subject is the headline. The length is the number of characters in the headline string. A channel the case does not cap, usually social, is not length-checked. The check asks for each requested channel at least once. It does not ask for a fixed number of variants, because the baseline's count is a setting and the planner chooses its own.

Banned words come from the brand profile. The match is a case-insensitive substring, the same rule as the example corpus. A required phrase uses that same match, across the headline, the body, and the call to action. A phrase split across two of those fields does not count.

## The judge rubric

`generate_tests` adds one `llm-rubric` to each case. The anchors come from that brand's profile, the same scale the critic uses. `defaultTest` stays the six deterministic checks. Promptfoo runs those checks and then this rubric (ADR 0029).

The assertion value is the anchored rubric, so the eval record shows the scale. The model does not see promptfoo's default grader prompt. `rubricPrompt` is `{{output}}`, the JSON row. `evals/providers/judge.py` parses that row and calls the gateway with `prompts/judge_v1.md`. The brand block and the five anchors are the cached prefix. The variant under review is the user text. One call per variant, so the judge has no neighbouring copy to prefer. The reply is a list of `{criterion, score}` items. `build_critique` checks that every rubric criterion appears once and that each score is 1 to 5. The model does not decide pass or fail, and it is not asked for fixes.

The row score is the mean of the per-criterion means, still on the 1–5 scale. The reason lists that mean, each criterion mean, and each variant's scores. The metric name is `judge`. A finished grade passes the assertion, including a mean of 1. The weight is 0, so the 1–5 mean stays out of the 0–1 deterministic average and still shows up as its own metric. Read `judge` for the voice comparison. A row that is not the eval JSON, a row with no variants, a reply that misses the rubric, or a gateway error fails the assertion. The rubric scored is the one on disk. When the row's `rubric_version` differs, the reason names both versions.

The judge's tokens and cost are on the grader result. They are not added to the row's usage, which stays the system under test. The judge call sits outside the graph, so it does not open a Langfuse run and it does not spend the pipeline's run budget.

Cross-check is off unless `BRANDFORGE_JUDGE_CROSSCHECK_MODEL` is a `provider:model` string and both of its prices are above 0. Then `generate_tests` adds a second `llm-rubric` with metric `judge_crosscheck`. That call still goes through the gateway as the judge tier, with the cross-check model and prices copied onto the tier for that call. The Opus cache-read override is not copied. Leave the model blank to score with the judge tier only. The three pipeline tiers are unchanged.

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

This runs `brightleaf_01_spring_blossom` through the baseline and the pipeline. That is two model-backed rows, and the pipeline row is several calls. After each row, the judge scores every variant on its own. A second model does the same when the cross-check is configured.

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

The output cell is the JSON row. The token and cost columns come from `tokenUsage` and `cost` on the provider, which is the system under test. The assertion columns are the deterministic checks and the `judge` metric. A failed check shows its reason. The judge reason starts with the 1–5 mean. The judge's own tokens and cost stay on that grader result.

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

`call_pipeline` is the graph. `generate_tests()` is the test list, and it does not call a model. `score_row` is the judge, and a real call uses the judge tier.

## Files

| Path | Role |
| --- | --- |
| `evals/promptfooconfig.yaml` | Providers, the case-id prompt, the test generator, the deterministic checks, cache off, concurrency 1. The judge rubric is added per case |
| `evals/providers/baseline.py` | Promptfoo `call_api` for the baseline |
| `evals/providers/pipeline.py` | Promptfoo `call_api` for the pipeline |
| `evals/providers/tests.py` | Promptfoo `generate_tests` |
| `evals/providers/assertions.py` | Promptfoo assertions. They forward to the package |
| `evals/providers/judge.py` | Promptfoo grader for the `llm-rubric`. It forwards to the package |
| `evals/cases/*.yaml` | The 30 cases (ADR 0026) |
| `src/brandforge/evals/promptfoo.py` | The row shape and the two provider functions |
| `src/brandforge/evals/cases.py` | Case loader |
| `src/brandforge/evals/assertions.py` | The deterministic checks (ADR 0028) |
| `src/brandforge/evals/judge.py` | The judge grader (ADR 0029) |
| `src/brandforge/prompts/judge_v1.md` | The versioned judge prompt. One variant, scores only |
| `tests/test_promptfoo_providers.py` | Provider tests with the model mocked |
| `tests/test_assertions.py` | Deterministic checks, with no model call |
| `tests/test_judge.py` | Judge grader tests. The gateway is faked |
| `docs/adr/0027_promptfoo_providers.md` | Why the prompt is a case id and why both systems share one row |
| `docs/adr/0028_deterministic_assertions.md` | What the deterministic checks measure, and what they leave to the judge |
| `docs/adr/0029_judge_rubric.md` | How the 1–5 mean is scored, and how the optional cross-check borrows the judge tier |
