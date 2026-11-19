# Risks and blockers

| id | risk | impact | mitigation | state |
| --- | --- | --- | --- | --- |
| R1 | Docker is not installed on the build host, so `docker compose up` cannot be verified locally. | "Fresh clone in <10 min" stays unverified. | Ask the owner before installing system packages. CI can build the image instead. | open |
| R2 | No Anthropic API key is in the credential catalog. The OpenRouter key exists ($5 limit, not free tier); free models have per-minute and per-day request caps, report $0 cost, and queue slowly. | Live evals and recordings are blocked until a provider is chosen. Cost and latency targets would not be comparable on free models. | The provider-neutral `ModelClient` and replay keep CI and lanes unblocked; the owner picks the provider at the M0 checkpoint. | open |
| R3 | GitHub refuses branch protection and rulesets on this free-plan private repo (HTTP 403: "Upgrade to GitHub Pro or make this repository public"). | Required PRs, required checks, and linear history are not enforced by GitHub. | Enforced: squash-only merges (merge commits and rebase disabled), delete branch on merge. Procedural: no direct pushes to `main` after bootstrap; the orchestrator merges only through the gate. Owner decides: upgrade to Pro, or accept procedural enforcement. | blocked (owner) |
| R4 | The `mcp` SDK is 2.x (newer than many examples). | Lane A API surprises. | Lane A reads the installed package's source before coding. | open |
