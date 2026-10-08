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

Each run also prints a Trace ID. Tracing is optional: put your Langfuse keys (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` if you are not on the cloud free tier) in `.env` and every run becomes one trace, with a span per node and a generation per model call. Without the keys, or with `BRANDFORGE_TRACING_ENABLED=false`, the run is untouched and the CLI says tracing is off.

A brief is a YAML file with `product`, `audience`, `objective` (`awareness`, `consideration` or `conversion`), `channels` (`search`, `social`, `display`, `email`) and optional `constraints`. Known brands: `brightleaf`, `ledgerly`, `voltride`.

## Project layout

```text
src/brandforge/
├── agents/        # planner, writer, critic, reviser, assembler
├── llm/           # LLM gateway and per-provider adapters: retries, token accounting
├── retrieval/     # brand example index and retriever
└── interfaces/    # CLI, API and demo page
```

## Roadmap

Work is tracked as [issues](https://github.com/DavidOwenBailey/brandforge/issues) grouped into [milestones](https://github.com/DavidOwenBailey/brandforge/milestones), from M0 Inception to M8 Release (`v1.0.0`).

## Licence

[MIT](LICENSE)
