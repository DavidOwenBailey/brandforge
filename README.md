# brandforge
Multi-agent pipeline that turns creative briefs into on-brand ad copy.

Project status: in active development.
Phase 1 delivery: 10 October 2026.

## What it does

A team of agents, orchestrated with LangGraph, plans, writes, critiques against a brand-voice rubric and revises ad copy variants. An evaluation suite measures whether the full pipeline beats a single-prompt baseline, and at what cost.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv installs the pinned Python version (3.12) for you.

```bash
git clone https://github.com/DavidOwenBailey/brandforge.git
cd brandforge
uv sync
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uv run brandforge generate --brand voltride --brief src/brandforge/briefs/voltride_01_commuter_ebike.yaml
```

Each run prints a Run ID, and its state is saved after every node. `uv run brandforge inspect <run-id>` shows the state after each node, and `--step N` prints the full state at one step. Checkpoints live in `.brandforge/checkpoints.sqlite` (change it with `BRANDFORGE_CHECKPOINT_DB`).

Each run also prints a Trace ID. Tracing is optional, and the default backend is Langfuse Cloud's free tier: put the project's `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` (`LANGFUSE_HOST` stays `https://cloud.langfuse.com`) and every run becomes one trace, with a span per node and a generation per model call. Without the keys, or with `BRANDFORGE_TRACING_ENABLED=false`, the run is untouched and the CLI says tracing is off. To keep traces on the laptop instead, start the optional stack in `deploy/langfuse/docker-compose.yml` and point `LANGFUSE_HOST` at it. The steps are in [docs/langfuse.md](docs/langfuse.md).

The same run writes structured logs to stderr: one JSON object per line, each carrying that run's ID (and the trace ID, when there is one), so a log line can be tied to the printed Run ID and to the Langfuse trace. `BRANDFORGE_LOG_LEVEL` sets the level (default `INFO`) and `BRANDFORGE_LOG_FORMAT=console` renders the same fields for a person. The tables the CLI prints stay on stdout.

A brief is a YAML file with `product`, `audience`, `objective` (`awareness`, `consideration` or `conversion`), `channels` (`search`, `social`, `display`, `email`) and optional `constraints`. Known brands: `brightleaf`, `ledgerly`, `voltride`.

`uv run brandforge index` embeds the approved examples for those brands. A generate run then retrieves the nearest examples for each channel. With no index, the run logs a warning and writes the copy without them. `BRANDFORGE_RETRIEVAL_ENABLED=false` skips that search on purpose, which is how an eval compares the pipeline with and without examples.

## Project layout

```text
src/brandforge/
├── agents/        # planner, retriever, writer, critic, reviser, assembler
├── llm/           # LLM gateway and per-provider adapters: retries, token accounting
├── retrieval/     # brand example index and retriever
└── interfaces/    # CLI, API and demo page
```

## Roadmap

Work is tracked as [issues](https://github.com/DavidOwenBailey/brandforge/issues) grouped into [milestones](https://github.com/DavidOwenBailey/brandforge/milestones), from M0 Inception to M8 Release (`v1.0.0`).

## Licence

[MIT](LICENSE)
