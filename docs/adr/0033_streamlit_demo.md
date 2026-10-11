# 0033. The demo page calls the API

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

BF-40 asks for a Streamlit page: pick a brand, paste a brief, and see the
variants, scores and flags. The page has to use the FastAPI endpoint from
0032. The architecture already chose Streamlit over Gradio for the demo, and
it left the page for this task. The API has no auth and binds to loopback.
CORS is not enabled.

## Decision

`brandforge demo` serves `interfaces/app.py` on `127.0.0.1` port `8501` unless
`BRANDFORGE_DEMO_HOST`, `BRANDFORGE_DEMO_PORT`, `--host` or `--port` says
otherwise. The page is a client of the API. It does not import the graph or
the gateway.

- **Brand ids** come from the brand store, the same YAML directory the API
  loads. There is no list route. The select defaults to `voltride`.
- **The brief is pasted.** The text area accepts a brief file's YAML, or the
  same fields as JSON. The page validates that text as a `Brief` and posts
  `{"brand", "brief"}` to `{BRANDFORGE_API_BASE_URL}/generate`. The default
  origin is `http://127.0.0.1:8000`. The origin has no path. A trailing slash
  is stripped. `0.0.0.0` can be a bind address, so the base URL is a separate
  setting from `BRANDFORGE_API_HOST`.
- **The page renders the `RunResult`.** Each variant shows its headline, body
  and call to action. A table shows the criterion scores, the overall score
  and whether the variant passed or is flagged. A variant that was never
  scored is `flagged (not scored)`, the same words as the CLI. Status, errors,
  cost and `trace_id` are shown. HTTP 200 with status `failed` is a finished
  run, as in 0032.
- **The page waits** for the wall-clock budget, plus one model-call timeout,
  plus 15 seconds. A call can start just before the budget ends.
- **The browser talks only to Streamlit.** The page calls the API with httpx
  from the Streamlit process, so CORS stays off. Headless mode is on, and
  Streamlit usage stats are off.
- **No authentication.** Loopback is the control, same as the API. Do not
  publish the port.

## Alternatives considered

- **Call `run_graph` inside the Streamlit process.** One process, and no need
  to start the API. That is a second way into the pipeline. The task is to use
  the endpoint, and the page would drift from what `POST /generate` returns.
- **Add `GET /brands`.** The dropdown would then be a pure client. The brand
  list is the YAML directory, and a new route would be a second contract for a
  select box. The page and the API both read those files.
- **Call the API from the browser.** The page would be static and would need
  CORS, which 0032 turns down. A server-side client keeps that decision.

## Consequences

- **Gained:** the demo shows the same result as the CLI and the API, including
  a failed run and its trace ID. `brandforge inspect` can open a run that
  started from the page, because the API writes the same checkpoint file.
  The page code does not call a model. Docker (BF-41) can point
  `BRANDFORGE_API_BASE_URL` at the API service.
- **Trade-off accepted:** the demo is two processes. The page says so when it
  cannot connect. Anyone who can reach either port can spend the model key.
  The page reads settings for the URL and the timeout, so a shared `.env`
  loads the key into the demo process as well. The page never sends that key.
