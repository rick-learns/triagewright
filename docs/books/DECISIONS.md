# Decisions (ADR index)

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [ADR-001](../adr/ADR-001-stack-and-architecture.md) | Stack and architecture | Accepted (council, 5 rounds) | 2026-11-19 |
| [ADR-002](../adr/ADR-002-gemini-free-tier-default-provider.md) | Gemini API free tier as the default live provider | Accepted (owner decision; #4), written up in #28 | 2026-11-19 |
| [ADR-003](../adr/ADR-003-pace-live-calls-per-quota.md) | Pace live model calls per quota and model, off deadlines and latency | Accepted (#18), written up in #28 | 2026-11-19 |
| [ADR-004](../adr/ADR-004-code-decides-refund-outcomes.md) | Code decides every refund outcome | Accepted (#19), written up in #28 | 2026-11-19 |
| [ADR-005](../adr/ADR-005-approval-queue-is-the-human-step.md) | The approval queue is the human step; reviewer escalations code already handled are left to code | Accepted (#20, extended by #21), written up in #28 | 2026-11-19 |
| [ADR-006](../adr/ADR-006-span-errors-without-exception-messages.md) | Span errors never carry exception messages | Accepted (#20), written up in #28 | 2026-11-19 |
| [ADR-007](../adr/ADR-007-rootless-podman-local-stack.md) | Rootless Podman replaces Docker for the local stack | Accepted (owner decision; recorded in #4), written up in #28 | 2026-11-19 |

## Owner decisions

| date | decision |
| --- | --- |
| 2026-11-19 | Live model provider: OpenRouter only, `:free` models only (R2). |
| 2026-11-19 | Accept procedural enforcement on `main` instead of GitHub Pro (R3). |
| 2026-11-19 | Install Docker on the build host. Blocked: no working sudo credential (R1). |
| 2026-11-19 | Add the Gemini API free tier as a live provider ($0, free plan, no billing). It's the default; OpenRouter `:free` is the fallback (R2). |
| 2026-11-19 | Use rootless Podman instead of Docker for the local stack (R1 resolved). |
| 2026-11-19 | Finish M2 and M3 without stopping. When a free model's quota runs out, use another free Gemini model or OpenRouter `:free`, and label every result with its provider and model. No paid models. |
| 2026-11-19 | A three-model council (GPT Sol, GPT Astra, Claude Opus; high effort; read-only) reviews the eval cases instead of the owner. Only accepted cases get `reviewed: true`, and it's recorded as a council review (#39). |
