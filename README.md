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
