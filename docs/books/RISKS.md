# Risks and blockers

| id | risk | impact | mitigation | state |
| --- | --- | --- | --- | --- |
| R1 | Docker is not installed on the build host, so `docker compose up` cannot be verified locally. | "Fresh clone in <10 min" stays unverified. | Ask the owner before installing system packages. CI can build the image instead. | open |
| R2 | No Anthropic API key is in the credential catalog. The OpenRouter key exists ($5 limit, not free tier); free models have per-minute and per-day request caps, report $0 cost, and queue slowly. | Live evals and recordings are blocked until a provider is chosen. Cost and latency targets would not be comparable on free models. | The provider-neutral `ModelClient` and replay keep CI and lanes unblocked; the owner picks the provider at the M0 checkpoint. | open |
| R3 | Branch protection on private repos may need a paid GitHub plan. | `main` may not be protectable. | Try rulesets and classic protection. If both are refused, ask the owner. | open |
| R4 | The `mcp` SDK is 2.x (newer than many examples). | Lane A API surprises. | Lane A reads the installed package's source before coding. | open |
