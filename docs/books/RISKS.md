# Risks and blockers

| id | risk | impact | mitigation | state |
| --- | --- | --- | --- | --- |
| R1 | Docker is not installed on the build host. | `docker compose up` couldn't be verified locally. | Use rootless Podman 4.9.3 with podman-compose 1.6.0, already installed. `podman build` built the image (27 s) and `podman-compose config` accepts `compose.yaml`. `podman compose up` is verified once lanes A and H land. | resolved (Podman) |
| R2 | Live models are free tiers only (owner decision): Gemini API free tier (default) and OpenRouter `:free` (20 requests/min, 1,000/day on this key). Free models report $0 and can return 429/503 under load. | Live p95 is noisy; "< $0.02/ticket" is met trivially. A Gemini key on a billed project would charge. | Settings and adapters reject non-free model ids. The Gemini key's project is on the free plan (owner confirmed in AI Studio, 2026-11-19). 429/503 are retryable; recording is throttled and resumable, after A, C, and D land. The report states provider, models, and mode. | accepted |
| R3 | GitHub refuses branch protection and rulesets on this free-plan private repo (HTTP 403). | Required PRs, required checks, and linear history are not enforced by GitHub. | Owner accepted procedural enforcement: no direct pushes to `main` after bootstrap, and the orchestrator merges only through the gate. Squash-only merges and branch deletion are enforced by repo settings. | accepted |
| R4 | The `mcp` SDK is 2.x (newer than many examples). | Lane A API surprises. | Lane A reads the installed package's source before coding. | open |
