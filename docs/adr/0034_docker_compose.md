# 0034. Docker Compose runs the API and the demo from one image

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

BF-41 asks for `docker compose up` to start the API and the demo page with one
API key. The API (0032) and the page (0033) are already two processes. The page
calls `POST /generate` and does not run the graph. Both bind to loopback on the
laptop, and neither has authentication.

ADR 0022 kept self-hosted Langfuse in its own Compose file so that a later
`docker compose up` would mean the app. The architecture names that root file.
The default model tiers are Anthropic, and a fresh clone is documented as one
key, `ANTHROPIC_API_KEY`.

## Decision

One image, built from the root `Dockerfile`, tagged `brandforge:local`. Compose
runs it twice. `api` runs `brandforge serve`. `demo` runs `brandforge demo`.
The image is not published and is not pulled. `pull_policy: build` keeps
Compose from looking the tag up on a registry.

- **Inside the Compose network the processes listen on `0.0.0.0`.** A listener
  on `127.0.0.1` inside a container accepts only that container. The published
  port would have nothing to connect to, and the demo container could not call
  the API. The commands pass `--host 0.0.0.0`. Compose also sets
  `BRANDFORGE_API_HOST` and `BRANDFORGE_DEMO_HOST`, so a later reader of those
  settings sees the address the process bound.
- **The published ports bind to `127.0.0.1` on the laptop.** API `8000`, demo
  `8501`. That is the same control as 0032 and 0033: do not put an unauthenticated
  route on the LAN. Open the page at `http://127.0.0.1:8501` and the docs at
  `http://127.0.0.1:8000/docs`.
- **The demo calls `http://api:8000`.** That hostname is the API service. The
  browser still talks only to Streamlit, and CORS stays off (0033). The demo
  waits until the API's healthcheck passes.
- **One `.env`, one key.** Both services read the repository `.env`. The file
  is required, and it is not copied into the image. The default tiers need
  `ANTHROPIC_API_KEY` only. A Gemini key is needed only when a tier uses one,
  the same rule as on the laptop. An empty key starts the servers and fails on
  the first brief, as the CLI does.
- **Compose replaces the laptop addresses and the data paths.** `.env` says
  `127.0.0.1` and a relative `.brandforge/` path, which are wrong in this
  network. The API writes checkpoints and the Chroma index to
  `/app/.brandforge` on a named volume. The demo does not mount that volume, so
  it cannot lock the SQLite file. `docker compose exec api brandforge inspect
  <run-id>` reads a run the API saved. A host `brandforge inspect` does not see
  it.
- **The image does not build the example index.** A missing index is an empty
  retrieval, which the pipeline already handles (0024). `docker compose run
  --rm api brandforge index` writes the index into the volume. That command
  downloads the local ONNX model into the container user's home. It does not
  need an API key (0023).
- **Langfuse stays a separate file (0022).** Cloud is the default and a
  container can reach it. `localhost` inside a container is the container. For
  the optional laptop stack, set `LANGFUSE_HOST` to
  `http://host.docker.internal:3000`. Compose adds that hostname. Tracing
  still does nothing when the keys are empty (0019).
- **The healthchecks do not call a model.** The API check fetches
  `/openapi.json`. The demo check fetches Streamlit's `/_stcore/health`.
- **The image runs as `brandforge`, not root.** The uv tag is `0.12.23`, the
  version that locked the repo, on Python 3.12. CI does not build or start the
  image. The tests check the files, as they do for the Langfuse stack.

## Alternatives considered

- **One container, both processes.** A supervisor, and a crash of one takes the
  other down. The page is already a client of a separate API. Two services is
  what Compose is for.
- **Publish `0.0.0.0` on the laptop.** The page would open as `localhost` on
  more machines, and anyone on the LAN could spend the key. Loopback is the
  control 0032 already chose.
- **Listen on `127.0.0.1` inside the container.** The published port arrives on
  the container's other interface, so the connection is refused. The demo
  container's `127.0.0.1` is not the API.
- **Copy `.env` into the image.** The key would sit in an image layer. Compose
  injects it at run time.
- **Pass only `ANTHROPIC_API_KEY`, and ignore the rest of `.env`.** The
  container would drop the laptop's budgets, tiers and Langfuse keys, and the
  page's timeout would no longer match the API. Sharing `.env` keeps one
  configuration. Compose overrides the few values that must differ.
- **Bind-mount the repository `.brandforge` directory.** A host `inspect` would
  see container runs without `exec`. SQLite on a Docker Desktop bind mount,
  especially from Windows, locks badly. The named volume avoids that.
- **Build the index during `docker build`.** The first brief would have
  examples, and the build would download the ONNX model into the image. The
  download is large, and an empty index is already a working run. The index
  stays an explicit command.
- **Add `GET /health`.** A new route for a check `/openapi.json` already
  serves, and that route does not run a brief.
- **Put Langfuse in this file under a profile.** Rejected in 0022. A profile is
  easy to turn on, and `docker compose up` would no longer mean the app.

## Consequences

- **Gained:** `docker compose up`, with one key in `.env`, serves the API and
  the page on this machine. The image contains no secret. The page shows the
  same result as `POST /generate`.
- **Trade-off accepted:** `.env.example` still says `127.0.0.1`, because that
  is correct for `uv run` on the laptop. Compose replaces it. The example file
  says so.
- **Trade-off accepted:** `brandforge inspect` on the host does not see a run
  that the container saved. `docker compose exec api brandforge inspect` does.
- **Trade-off accepted:** CI does not prove the image boots. The file tests
  guard the contract: loopback ports, one shared `.env`, no model call in the
  healthchecks, Langfuse left out.
- **Trade-off accepted:** the demo process receives the API key, because it
  loads the same `.env` (0033). The page does not send the key.
- **Trade-off accepted:** moving uv is a deliberate tag bump. The project's
  build backend is still pinned below `0.9`, so this uv downloads that backend
  instead of using a newer bundled copy.
