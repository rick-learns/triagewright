# ADR-004: Code decides every refund outcome

- Status: Accepted (PR #19)
- Date: 2026-11-19

## Context

ADR-001 put the refund policy in code (`contracts.policy.evaluate_refund`) and
the write path in the executor, but the coordinator model still chose whether
to send a refund through that path. On Gemini Flash-Lite the coordinator
answered refund tickets itself and never had a refund proposed. The prompt
said to use the `action` subagent for refunds; the model didn't follow it.
Its replies could deny or promise a refund that the policy never evaluated.
Separately, the model writes its reply before the executor runs, so any
outcome it states is a guess.

## Decision

- **What counts as a refund ask.** Triage's `refund_request` category, or its
  `refund_requested` flag for a money-back ask filed under another category
  (`contracts.agents.TriageResult`). Refund-policy questions and status
  checks on earlier refunds don't count.
- **A gate on finalize.** For a refund ask, `Coordinator._check_final` rejects
  a `finalize` call unless the reply cites `kb_refund_policy` with a verified
  quote and the outcome is left to code (`_refund_decidable`): an
  `issue_refund` proposal is queued, or the ticket has no customer, or
  injection is already suspected, or the draft is `needs_info` and the order
  is unclear. The order is unclear when the ticket names no order id and the
  listed orders don't narrow to one; a full page of orders never narrows to
  one, since older orders may match too. The model gets one repair
  attempt; a second failure is the safe fallback. On a refund ask, the
  coordinator's own read tools shrink to `search_kb`, so order lookups and the
  proposal go through the action subagent.
- **The executor decides.** `GuardedActionExecutor.propose` maps an
  approval that already exists for the same proposal from its stored status
  (`pending` and `approved` to `pending_approval`, `rejected` to `blocked`,
  `executed` to `skipped`);
  otherwise it fetches eligibility itself, returns `failed` if the lookup
  fails, and submits an eligible refund for approval or blocks it. For a
  refund blocked with eligibility reasons, the coordinator applies
  `policy.REFUND_OUTCOME[eligibility.reasons[0]]`: refusal or a queue.
- **Code writes the outcome text.** Once a refund is proposed, the model's
  customer reply is replaced by a fixed lead-in (`prompts.REFUND_OUTCOME_REPLY`) and
  the per-outcome notes from `Coordinator._note`.
- **Only two replies without a proposal,** both written by code
  (`Coordinator._own_refund_reply`): a sign-in request when the ticket has no
  customer, or "which order?" when the ticket names no order and the
  customer's complete order list doesn't narrow to one by a named product or
  by count.
- **Injection.** A refund ask with injection suspected anywhere in the request,
  including the reviewer's screen, goes to the security queue, and its
  customer reply is the fixed `prompts.FALLBACK_REPLY`, not model text.
- **Refusals.** If the action subagent refuses, the ticket is handed off to
  tier1 rather than told "refused".
- The behavior is part of the contract in `docs/contracts/behavior.md`
  ("Refund decisions").

## Consequences

- No model can deny, approve, or settle a refund. Every refund reply follows
  the policy table, and evals can check the outcome from the call log and the
  approval list.
- Live on 10 `happy-*` cases (Gemini Flash-Lite, with ADR-003 pacing): pass
  rate rose from 60% to 70%, with 0 unsafe actions, and happy-005 now
  reaches `pending_approval`.
- Refund replies are less conversational: the model's wording is dropped once
  an outcome exists.
- The gate depends on triage spotting the refund ask. A missed ask falls back
  to the ordinary flow, where the executor still guards any write.
- A side fix: `propose_action`'s schema narrows `arguments` to
  `RefundProposal`, because Gemini rejected the recursive JSON schema with
  HTTP 400.

## Alternatives

- **A stronger prompt rule.** Rejected: the prompt rule was already there and
  was skipped. A deterministic gate is needed where compliance matters.
- **Let the model write the reply after the executor runs.** Rejected: one
  more model call, and the model could still misstate the outcome.
- **Filter the model's reply for refund claims.** Rejected: phrasing varies too
  much to filter reliably; replacing the text is simpler and certain.
