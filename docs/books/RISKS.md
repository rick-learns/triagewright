# Risks and blockers

| id | risk | impact | mitigation | state |
| --- | --- | --- | --- | --- |
| R1 | Docker is not installed on the build host, so `docker compose up` cannot be verified locally. | "Fresh clone in <10 min" stays unverified. | The owner approved installing Docker, but the catalogued sudo credential was rejected and the owner declined a sudo prompt. Compose and the Dockerfile still ship; a CI image build can stand in. | blocked (sudo) |
| R2 | The only live provider is OpenRouter `:free` models (owner decision). Free models are capped (20 requests/min; 50 or 1,000 requests/day depending on lifetime credit), report $0 cost, and queue slowly. | Recording all cases can take several days. "< $0.02/ticket" is met trivially, and live p95 is noisy. | Record only after A, C, and D land (one pass). Throttled, resumable recording. The report states provider, models, and mode. | accepted |
| R3 | GitHub refuses branch protection and rulesets on this free-plan private repo (HTTP 403). | Required PRs, required checks, and linear history are not enforced by GitHub. | Owner accepted procedural enforcement: no direct pushes to `main` after bootstrap, and the orchestrator merges only through the gate. Squash-only merges and branch deletion are enforced by repo settings. | accepted |
| R4 | The `mcp` SDK is 2.x (newer than many examples). | Lane A API surprises. | Lane A reads the installed package's source before coding. | open |
