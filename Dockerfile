# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_FROZEN=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-install-project
COPY src ./src
COPY data ./data
COPY config ./config
RUN uv sync --no-dev

RUN useradd --create-home --uid 10001 app && mkdir -p /app/var && chown app /app/var
USER app
ENV PATH="/app/.venv/bin:$PATH" TW_API_HOST=0.0.0.0
EXPOSE 8080 8765
CMD ["triagewright-api"]
