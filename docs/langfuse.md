# Langfuse

Tracing is optional. The default backend is [Langfuse Cloud](https://cloud.langfuse.com)'s free tier. A self-hosted stack is available when prompts should stay on the laptop. BrandForge sends the same spans either way (ADR 0019). Nothing starts Langfuse for you, and a run with no keys is unchanged.

## Cloud (default)

1. Create a project at <https://cloud.langfuse.com>. The free tier needs no card.
2. Copy the project's public and secret keys into the repository-root `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

3. Leave `BRANDFORGE_TRACING_ENABLED=true`.
4. Run `uv run brandforge generate`. The CLI prints a trace ID. Open that trace in the Cloud UI.

Leave either key empty, or set `BRANDFORGE_TRACING_ENABLED=false`, and the run is not traced.

## Self-hosted (optional)

Requires Docker Desktop, with a few GB free. ClickHouse is the heavy container. The stack is Langfuse 4.55.0 plus Postgres, ClickHouse, Redis and MinIO, defined in `deploy/langfuse/docker-compose.yml`. It is local-only: no high availability, and the passwords in that file are development defaults (ADR 0022).

From the repository root:

```bash
docker compose -f deploy/langfuse/docker-compose.yml up
```

The first start downloads the images and then takes a couple of minutes. Wait until the `langfuse-web` container logs `Ready`, then open <http://localhost:3000>. Langfuse ships with no user. Choose **Sign up**, create an organization and a project, and copy that project's keys.

Point the repository-root `.env` at the local UI:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

`LANGFUSE_HOST` has to be the URL the browser uses. The UI is bound to `127.0.0.1`, so another machine cannot open it. To use a different host port, set `LANGFUSE_WEB_PORT` and `NEXTAUTH_URL` together in `deploy/langfuse/.env` (copy from `deploy/langfuse/.env.example`) and use that same URL as `LANGFUSE_HOST`. MinIO is the same kind of pair: `MINIO_PORT` and `LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT`. Text traces do not need MinIO's published port.

Stop the stack with:

```bash
docker compose -f deploy/langfuse/docker-compose.yml down
```

Add `-v` only when you mean to delete the traces. Volumes keep the data across a plain `down`.

### What is published

| Port | Service | Bound to | Why |
| --- | --- | --- | --- |
| 3000 | Langfuse UI and API | 127.0.0.1 | The SDK and the browser |
| 9090 | MinIO | 127.0.0.1 | Browser media uploads |

Postgres, Redis and ClickHouse stay on the Docker network. They are not published on the host, so they do not clash with a local database.

### Headless initialization

To create the user, organization, project and API keys on startup, uncomment the `LANGFUSE_INIT_*` block in `deploy/langfuse/.env`. The keys you set there are the keys you put in the repository-root `.env`. Do not quote the values. With the block left commented, the stack has no built-in login.

### Upgrading

The Langfuse image tags are pinned, and the web and worker tags are the same version. To move to a newer patch, change both tags in `deploy/langfuse/docker-compose.yml`, then run `docker compose -f deploy/langfuse/docker-compose.yml up --pull always`. MinIO is the one unpinned image: Chainguard's image is what the healthcheck expects, and it has no semver tag.

Telemetry to Langfuse is off (`TELEMETRY_ENABLED=false`).
