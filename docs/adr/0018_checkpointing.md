# 0018. Checkpointing: state saved to SQLite after every node, read back with `inspect`

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

The design says LangGraph's SQLite checkpointer stores state after each node, so a failed run
can be inspected (0001). Until now the graph was compiled without one and a run's state lived
only in memory. A run that ended `partial` or `failed` left behind its result and nothing else,
and a run that was stopped part-way (Ctrl+C, a crash) left nothing at all. `checkpoint_db` has
been in config since BF-04, unused.

## Decision

The graph can be compiled with a checkpointer, and `brandforge generate` always uses one.

- **The store** is a SQLite file at `settings.checkpoint_db` (default
  `.brandforge/checkpoints.sqlite`, already git-ignored), per the local-first decision (0007).
  It needs the separate `langgraph-checkpoint-sqlite` package.
- **The thread ID is the run's `run_id`.** `build_graph` and `run_graph` take an optional
  `checkpointer`; with one, the whole state is saved after every node. Without one nothing is
  written, which is what the evals and most tests want. `generate` now prints the run ID.
- **`brandforge/checkpointing.py`** has `open_checkpointer` (opens and closes the file, makes
  the folder if needed) and `load_run_steps` (every checkpoint of a run, oldest first).
- **`brandforge inspect <run_id>`** prints one row per checkpoint: the node that just ran, the
  variant and critique counts, revisions, errors, cumulative tokens and cost, the status, and
  which node was due next. A run that stopped early ends on a row that still has a next node.
  Errors recorded by the end of the run are listed under the table. `--step N` prints the whole
  state after one step as JSON.
- **The node that produced a checkpoint** is read from the checkpoint before it (the node that
  was due to run next there), because LangGraph's checkpoint metadata does not record it.
  LangGraph's own first checkpoint holds only the raw input, so it is not shown; step 0 is the
  starting state.
- **The saver is given an allowlist of every model in `brandforge.models`.** State holds
  Pydantic models directly. LangGraph warns when it rebuilds a type it was not told to trust
  and says it will block it in a future version. The list is taken from the module, so a new
  contract is allowed without anyone remembering to add it.
- **A `run_id` that already has checkpoints is refused.** A second run on the same thread would
  start from the first one's final state, and the usage and error reducers would add to its
  totals. Run IDs are random by default, so this only catches a reused one.
- **Reading never creates a database.** `inspect` on a missing file says so and exits 1, rather
  than leaving an empty file behind.

## Alternatives considered

- **`MemorySaver`:** nothing survives the process, so a run that crashed could not be inspected
  afterwards, and `inspect` would have nothing to read.
- **Write our own JSON dump of state in the node guard:** would work for inspecting, but it
  reinvents what the checkpointer does and gives no path to resuming a run (BF-24 is the
  foundation for that).
- **A Postgres checkpointer:** the production choice (0007), not needed for a laptop.
- **Checkpoint inside `run_graph` whenever settings say so:** every eval run and test would
  write to disk and each would need isolating. An explicit argument keeps the default silent.
- **Record the node name in state:** a new field every node would have to set. Reading it from
  the previous checkpoint's `next` uses only LangGraph's public API.
- **Leave deserialization permissive:** works today, but logs a warning for every model type and
  stops working when LangGraph blocks unlisted types.

## Consequences

- **Gained:** any run, including one that was killed, can be inspected by its run ID, and its
  state at every step is there to compare. It is also what resuming a run will build on.
- **Trade-off accepted:** every checkpoint holds the whole state, including the brand profile
  and rubric, so each one repeats it. A nine-step run with two revisions came to about 40 KB of
  checkpoint data (the file with its SQLite overhead was about 90 KB). Nothing prunes old runs.
- **Trade-off accepted:** the brief and the generated copy are stored unencrypted on disk. That
  is fine here (local file, fictional brands, git-ignored) but production would need to decide
  retention and access.
- **Trade-off accepted:** `inspect` needs the run ID, and there is no command to list runs.
  `generate` prints it, and it is the file's `thread_id` column if one is lost.
- **Trade-off accepted:** `inspect` reads the node name from the checkpoint before, so if the
  graph ever runs nodes in parallel the `After` column would show them joined with commas
  rather than separate rows.
- **Not covered here:** resuming a run. A resumed run would keep its original `started_at`, so
  one resumed after the time limit would stop revising at once (0017); that needs deciding
  when resuming is built.
- **Not covered here:** the trace ID joins checkpoints and traces when tracing lands (BF-25).
