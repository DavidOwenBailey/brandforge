# 0027. promptfoo runs the baseline and the pipeline through one JSON row

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

The eval asks whether the pipeline beats a single prompt on the 30 cases from
0026. The architecture names promptfoo as the runner. promptfoo needs a provider
per system and a test list. The checks (length, banned words, a call to action,
the judge rubric) are the next tasks, so they need one output shape to score.
The model prompts already live in versioned files. promptfoo also has a prompt
field, and that is a different job: it only has to tell the provider which case
to run.

## Decision

`evals/promptfooconfig.yaml` lists two Python providers and a test generator.
`npx promptfoo eval` runs every case through both.

- The scripts under `evals/providers/` are the files promptfoo loads. Each one
  calls into `brandforge.evals.promptfoo`, which is what the tests and mypy
  cover. promptfoo resolves `file://` from the config directory, so the scripts
  sit next to the config.
- `generate_tests` emits one test per case, in filename order. `vars.case_id`
  is the id. The prompt template is `{{case_id}}`. A provider reads the var,
  and falls back to the rendered prompt when the var is missing. It then loads
  the case and calls `generate_baseline` or `run_graph`. It does not build a
  model prompt.
- `config["case_ids"]` on the generator keeps only those ids. The committed
  config does not set it, so the full set runs.
- Both providers return the same JSON object (`EvalOutput`): system, case,
  brand and rubric versions, status, variants, errors, tokens and cost. The
  baseline's status is `complete`. Its `flagged`, `run_id` and `trace_id` are
  null, because it has no critic and it is not a graph run. A pipeline row
  copies the `RunResult`, including `partial` and `failed`.
- promptfoo's `tokenUsage.prompt` is input plus cache write plus cache read.
  `completion` is output tokens. `cost` is `cost_usd`. The baseline sets
  `numRequests` to 1. The pipeline omits it: node usage is summed across
  revision passes, so a count taken from that list would be short. The token
  totals and the cost are exact.
- A case that cannot be run returns `{"error": "..."}` and does not raise, so
  one bad row leaves the rest of the run, and the other provider, intact.
- The pipeline provider does not pass a checkpointer. An eval row is the
  result, not a resumable run. Tracing follows settings, as it does for the CLI.
- The config turns promptfoo's cache off and sets concurrency to 1. The
  pipeline worker timeout is ten minutes. No new prompt file and no new
  setting.

## Alternatives considered

- **One provider script, with the system name in a test var.** One column, and
  a matrix inside the script. The comparison is supposed to be two providers
  side by side, and promptfoo's cache key includes the provider id. Two ids
  keep a baseline row from being served as a pipeline row.
- **Return a `RunResult` for the pipeline and a bare variant list for the
  baseline.** The next assertions would need two parsers for one comparison.
- **Let the promptfoo prompt be the model prompt.** That bypasses the versioned
  prompt files and the gateway, so the eval would no longer be running the
  systems the CLI runs.
- **Raise when a case fails.** promptfoo restarts a worker that crashes. One
  bad case would drop the rest of that provider's open calls.
- **Set the pipeline's `numRequests` from `len(usage.nodes)`.** A second critic
  pass is merged into the first, so the number would be low and look precise.
- **Put the providers in the package and point promptfoo at `src/`.** The
  architecture puts the config in `evals/`. The scripts stay there; the logic
  stays where the tests are.
- **Depend on promptfoo from Python.** promptfoo is a Node tool. The providers
  are plain functions that tool imports.
- **Leave the disk cache on.** The cache key covers the promptfoo prompt, which
  is only the case id. Editing `prompts/baseline_v2.md` would not bust it, and
  a second eval could report stale copy.

## Consequences

- **Gained:** `promptfoo eval` runs the baseline and the pipeline on the 30
  cases. A later assertion can treat `output` as one JSON shape.
- **Trade-off accepted:** the promptfoo prompt is not the text the model sees.
  The versioned prompt files still are.
- **Trade-off accepted:** `flagged` is null on baseline rows. A flagged-rate
  column is not comparable until a later check treats null as "not scored".
- **Trade-off accepted:** promptfoo has to run with the project interpreter,
  because that is the one that can import `brandforge`. Set `PROMPTFOO_PYTHON`
  when detection picks another one. The path is not pinned in the yaml:
  Windows and POSIX virtualenvs differ.
- **Trade-off accepted:** a full eval makes real model calls. The unit tests
  mock `generate_baseline` and `run_graph`. This task does not add promptfoo
  to CI; the smoke subset is a later task.
