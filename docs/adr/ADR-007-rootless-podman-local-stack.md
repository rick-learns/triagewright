# ADR-007: Rootless Podman replaces Docker for the local stack

- Status: Accepted (owner decision, 2026-11-19; recorded in PR #4)
- Date: 2026-11-19

## Context

The project must run on one machine with one compose command (ADR-001 §9).
Docker was not installed on the build host, and installing it was blocked: no
working sudo credential was available (risk R1 in `docs/books/RISKS.md`).
Rootless Podman 4.9.3 and podman-compose 1.6.0 were already installed and need
no root.

## Decision

- The supported local runtime is rootless Podman with podman-compose:
  `podman compose up`, and `podman compose --profile tracing up` for Jaeger.
- `compose.yaml` and `Dockerfile` stay runtime-neutral, so
  `docker compose up` works with the same files.
- The image runs as a non-root user (uid 10001). Published ports bind to
  `127.0.0.1` only (`api` on 8080, Jaeger on 16686 and 4318); the `mcp`
  service is not published and is reached only on the compose network.
- `README.md` documents `podman compose up` first.

## Consequences

- The stack runs without root or a system daemon, which also narrows what a
  container escape could reach.
- `podman build` built the image, and `podman-compose config` accepts
  `compose.yaml` (R1). A full `podman compose up` run was planned for after
  the MCP and API lanes landed.
- Compose features that podman-compose doesn't support must be avoided, so
  `compose.yaml` keeps to the common subset.

## Alternatives

- **Install Docker.** Blocked: no sudo credential on the build host.
- **No containers; run the processes directly.** Rejected as the only path:
  the goal is a one-command local stack. Direct runs still work for
  development (`triagewright-mcp`, `triagewright-api`).
- **Rootful Podman.** Rejected: it needs root and gains nothing here.
