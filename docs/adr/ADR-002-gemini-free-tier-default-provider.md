# ADR-002: Gemini API free tier as the default live provider

- Status: Accepted (owner decision, 2026-11-19; PR #4)
- Date: 2026-11-19

## Context

The project runs on free models only, at $0 (ADR-001 §2). The first live
provider was OpenRouter `:free`, and its shared free models often returned 503
or 429. Lanes needed a free provider they could count on before they started.
A Gemini API key on a project with no billing linked is free: going over a
limit returns 429 and never a charge. A probe on 2026-11-19 got correct tool
calls from Flash-Lite in 0.4–0.7 s, while Flash and Gemma returned 503 "high
demand".

## Decision

- `Provider.GEMINI` (`src/triagewright/contracts/llm.py`) is the preferred
  live provider, and the one recordings are planned with. `Provider.OPENROUTER`
  with `:free` models is the fallback. Both live adapters can record.
  `Provider.REPLAY` stays the default provider overall, so nothing calls the
  network unless `TW_LLM_PROVIDER` selects a live provider (in the environment
  or `.env`). Default replay serves eval-case recordings only.
- There is no paid fallback. `config.is_free_model(provider, model)` decides
  what is allowed: on Gemini, numbered Flash, Flash-Lite, and Gemma ids, with
  no `-latest` aliases, no Pro, and previews only from an explicit list
  (`_GEMINI_FREE_PREVIEWS`); on OpenRouter, ids ending in `:free`. `Settings`
  rejects any other id, with no opt-out, and both live adapters
  (`triagewright.llm.openai_compat`) refuse one too. This is model-id
  validation only: code can't tell whether a Gemini key's project has billing
  linked, so the key must belong to a project on the free plan (risk R2 in
  `docs/books/RISKS.md`).
- Defaults per role are in `config.DEFAULT_MODELS`: `gemini-3.5-flash-lite`
  for every agent role and `gemini-3.8-flash` for the judge; on OpenRouter,
  `google/gemma-4-31b-it:free` and `nvidia/nemotron-3-ultra-550b-a55b:free`.
  `TW_MODEL_*` overrides them per role. Replay uses the Gemini table.
- Switching to the fallback is a configuration change
  (`TW_LLM_PROVIDER=openrouter`), not an automatic failover inside a request.
- 429 and 503 are retryable in the adapter.

## Consequences

- Live spend stays at $0 as long as the Gemini key's project has no billing
  linked, and a non-free model id fails at startup rather than charging.
- Free-tier quotas are small (15 requests per minute per model for
  Flash-Lite), so live runs need pacing (ADR-003), and a full eval run can
  exhaust the daily quota on both providers.
- Free-tier prompts may be used by the provider for training. All data in the
  repo is synthetic.
- The model id is part of every recording fingerprint, so changing a default
  means re-recording.
- Because the contract is shaped like the Anthropic Messages API, a paid
  provider would be an adapter and a config change, but the settings check
  would have to change too, deliberately.

## Alternatives

- **OpenRouter `:free` only.** Rejected as the default: its shared free models
  were often overloaded. Kept as the fallback.
- **A paid fallback when free quotas run out.** Rejected: the owner allows $0
  only, and a silent paid fallback would break that.
- **Automatic failover between providers within a request.** Not adopted:
  model ids differ per provider, so recordings and fingerprints would no
  longer match, and live latency would mix two providers.
