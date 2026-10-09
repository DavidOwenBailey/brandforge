# 0022. Langfuse Cloud by default, self-hosted Compose kept optional

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

ADR 0019 sends every traced run to Langfuse, and it stays silent when the keys are absent. The
architecture allows two backends for that SDK: Langfuse Cloud's free tier, or Langfuse
self-hosted with Docker Compose. The build plan's done-when for BF-28 is that the self-hosted
option is documented and the cloud free tier remains the default.

Langfuse v4 is not one container. The current upstream Compose file runs a web container, a
worker, Postgres, ClickHouse, Redis and MinIO, and it is several hundred lines of optional
enterprise settings. The repository layout in the architecture puts "app + optional Langfuse"
in one root `docker-compose.yml`. That file does not exist yet: the app containers are BF-41,
whose done-when is that `docker compose up` starts the API and the demo page.

## Decision

**Cloud is the default.** `LANGFUSE_HOST` defaults to `https://cloud.langfuse.com`, in
`config.py` and in `.env.example`. A fresh clone traces nowhere until someone pastes a
project's keys. Those keys can be a Cloud project's. No container starts as part of
`brandforge generate`.

**Self-hosting is a second Compose file, not the app's.** `deploy/langfuse/docker-compose.yml`
is the whole local stack. It is started by name (`docker compose -f
deploy/langfuse/docker-compose.yml up`) and is never the file a later `docker compose up` in
the repository root will use for the API. The steps, including how the two `.env` files differ,
are in `docs/langfuse.md`.

**The file is a trimmed, pinned copy of upstream's local stack, not a clone of their
repository.** Web and worker are `4.55.0` (the release current when this was written).
ClickHouse is `25.12`, Redis `7.4`, Postgres `17`, matching the versions in Langfuse's own
Compose file at that release. MinIO stays on Chainguard's untagged image because that is the
image their healthcheck (`mc ready local`) is written for, and it publishes no semver tag.
Environment variables that exist only for Langfuse's in-app agent and enterprise features are
left out. Headless initialization is wired through but blank, so the stack has no built-in
login; signing up at `http://localhost:3000` is the documented path.

**The stack is local-only.** The UI and MinIO bind to `127.0.0.1`. Postgres, Redis and
ClickHouse are not published on the host. Passwords are development defaults, marked
`CHANGEME`, and overridable from `deploy/langfuse/.env`, which is git-ignored. Telemetry from
the Langfuse containers to Langfuse is off. Container logs rotate at 10 MB.

## Alternatives considered

- **One root `docker-compose.yml` with a `langfuse` profile, as the architecture sketch
  implied:** `docker compose up` would then be one command for everything, but BF-41's
  done-when is that the same command starts the API and the demo page. A profile is easy to
  turn on by accident, and a six-container observability stack should not sit on the app's
  critical path. A separate file keeps that promise.
- **Document "clone github.com/langfuse/langfuse and run its Compose file" and ship no file:**
  zero drift from upstream, and nothing to maintain. It also fails the task, which is a
  Compose file in this repository, and it asks a reviewer to trust a floating `:4` tag and a
  tree we do not pin.
- **Vendor Langfuse's Compose file unchanged:** stays closer to upstream, and copies several
  hundred lines of empty enterprise settings that this project will not set. A short file we
  can read is the better review.
- **Langfuse v2 (Postgres and one container):** smaller, and no longer the server the v4 SDK
  is developed against. The Python dependency is `langfuse` 4.x.
- **Turn headless initialization on, with keys committed in the example:** a run would trace
  on the first `docker compose up` with no signup. It would also commit a password and a
  working API key. Signup in the UI is one extra step and keeps secrets out of git.

## Consequences

- **Gained:** the default path is still "paste two Cloud keys", which is the path a fresh
  clone can finish without Docker. Someone who does not want prompts to leave the machine has
  a pinned stack and a short doc.
- **Gained:** BF-41 can add a root Compose file later without inheriting Langfuse, and
  `docker compose up` will mean the app.
- **Trade-off accepted:** the Compose file will drift from upstream. Upgrading is a deliberate
  tag bump of web and worker together, not a `git pull` of Langfuse.
- **Trade-off accepted:** MinIO is unpinned. A bad pull of that image is the one moving part.
- **Trade-off accepted:** the development passwords and the all-zero encryption key are in
  git. They protect nothing against someone who can read the Docker volumes. The bindings
  stop the UI being reached from the LAN. This stack is not a production deployment, and the
  doc says so.
- **Trade-off accepted:** self-hosting needs Docker Desktop and enough memory for ClickHouse.
  CI does not start the stack. The tests check the file's shape, the pinned tags, and that
  the cloud host is still the default.
