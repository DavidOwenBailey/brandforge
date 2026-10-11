# 0007. Keep the proof of concept on the laptop: YAML, SQLite and Chroma

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

The proof of concept has to run from a clone, with one API key, and a brand
change has to be visible in a pull request. A hosted database, a vector service
or a login would make the first run an infrastructure task. The same choice has
to leave a clear swap for production: Postgres for checkpoints and brand data,
pgvector for examples, and authentication on the API.

## Decision

Persistent brand data and run state stay in the repo or in git-ignored files on
the machine. The pipeline does not require a database server. There is one user
and one machine.

- **Brands** are one YAML file each in `src/brandforge/brands/`, loaded into
  `BrandProfile` (0006). Git is the history. There is one current version of
  each brand. An edit bumps `version`, and `rubric.version` when the rubric
  changed. `RunResult` records both. An older run is reproduced by checking out
  that commit. Two versions of one brand are not loaded in the same process.
  The loader and the file shape are in `docs/brand_store.md`.
- **Approved examples** are YAML in `src/brandforge/retrieval/examples/`.
  `brandforge index` writes one Chroma collection per brand under
  `.brandforge/chroma` (0023, 0024). The embedding model is Chroma's local ONNX
  build of MiniLM. The architecture named provider embeddings. Anthropic, the
  default provider, has no embeddings API, and a provider call would add a
  second key to a command that otherwise needs none. That choice is 0023.
- **Checkpoints** are a SQLite file at `.brandforge/checkpoints.sqlite` (0018).
  `brandforge generate`, `brandforge serve` and the demo's API use it. Eval
  runs and most tests compile the graph with no checkpointer, so they do not
  write the file.
- **Eval cases and the calibration sample** are YAML under `evals/`. The
  summary of a manual run is the eval artifact that gets committed (0031).
  Promptfoo's raw export stays out of git.
- **The API and the demo** listen on loopback and have no authentication
  (0032, 0033). Docker publishes those ports on `127.0.0.1` (0034).
- **`.brandforge/`**, the Chroma directory, SQLite files and `.env` are
  git-ignored. `.env.example` is the committed list of settings.

Tracing is the exception on purpose. The default backend is Langfuse Cloud's
free tier. With either key empty, the run is not traced and nothing is sent.
The self-hosted stack is `deploy/langfuse/docker-compose.yml`, and nothing
starts it (0022). A trace can contain prompt text. It is not the record of
brands, examples or checkpoints.

Not built, and named so a later change has somewhere to go: a Postgres
checkpointer, brand profiles in Postgres with `load_brand(id, version=...)`,
pgvector, authentication, and a tenant boundary per brand.

## Alternatives considered

- **Postgres and pgvector from the first milestone.** That is the production
  shape. A clone would need a database before the first brief. Rejected for
  the proof of concept.
- **Provider embeddings,** as the architecture's retrieval row says. The
  default provider cannot embed, so `brandforge index` would need a second
  key. Rejected (0023). The embedding function is injected, so a later
  comparison can swap it.
- **A database of brand versions,** so two versions load side by side. Worth
  it when more than one editor shares a process. For three fictional brands, a
  YAML diff is the review. Rejected until there is a tenant.
- **Tracing local by default.** Prompt text would stay on the laptop, and every
  clone would run ClickHouse before a traced generate. Cloud, with missing
  keys meaning off, is the default (0022).

## Consequences

- **Gained:** `uv sync`, one `ANTHROPIC_API_KEY` and `brandforge generate` are
  enough to produce copy. `brandforge index` needs no key. A brand or example
  change is a pull request. A result names the brand version and the rubric
  version that produced it.
- **Trade-off accepted:** one user on one machine. The API is not safe on a
  shared network. Checkpoints and the index do not follow the user to another
  computer. They are rebuilt or left behind.
- **Trade-off accepted:** the first real index downloads the MiniLM ONNX model
  into the user cache. After that download, indexing needs no network.
- **Trade-off accepted:** reproducing an old brand means checking out an old
  commit. The loader has no version argument.
- **Trade-off accepted:** with both Langfuse keys set, prompt text is sent to
  Langfuse Cloud. Leave a key empty to keep the run on the machine.
