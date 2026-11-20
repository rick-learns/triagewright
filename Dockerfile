# syntax=docker/dockerfile:1
# Build: resolve the locked dependencies, then install the project (not editable) into /app/.venv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_FROZEN=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-install-project
COPY src ./src
RUN uv sync --no-dev --no-editable

# Runtime: the venv and the data the app reads relative to /app. No uv, no sources, no build tools.
FROM python:3.12-slim
# Debian security fixes land before the base image is rebuilt (zlib CVE-2026-85091, 2026-11).
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY data ./data
COPY config ./config
# var/ holds the SQLite files and traces; compose mounts a volume here.
RUN mkdir -p var && chown app:app var
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TW_API_HOST=0.0.0.0
EXPOSE 8080 8765
CMD ["triagewright-api"]
