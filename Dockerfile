# syntax=docker/dockerfile:1
# BrandForge API and demo page (BF-41, ADR 0034).
# One image. Compose runs it twice: `brandforge serve` and `brandforge demo`.
# The uv tag matches the uv that locked this repo. Do not float the tag.

FROM ghcr.io/astral-sh/uv:0.12.23-python3.12-trixie-slim

# Chroma's local ONNX embeddings load OpenMP. The slim image does not ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1000 brandforge \
    && useradd --system --gid 1000 --uid 1000 --create-home brandforge

WORKDIR /app

# Logs flush a line at a time. Bytecode is compiled at build time.
# copy: the uv cache is a mount, so hard links would point at another filesystem.
# The image Python is 3.12. Do not download another interpreter.
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

# Dependencies first, so a source change does not download them again.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# .dockerignore keeps .env, .venv and .brandforge out of this copy.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

ENV PATH="/app/.venv/bin:$PATH"

# A new named volume copies this directory, including its owner. The API writes
# checkpoints and the example index here.
RUN mkdir -p /app/.brandforge \
    && chown brandforge:brandforge /app/.brandforge

# The uv image sets an entrypoint of `uv`. The process we want is brandforge.
ENTRYPOINT []

USER brandforge

EXPOSE 8000
CMD ["brandforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
