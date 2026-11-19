# ADR-005: The approval queue is the human step; reviewer escalations code already handled are left to code

- Status: Accepted (PR #20, extended by PR #21)
- Date: 2026-11-19

## Context

The escalation reviewer subagent runs before every finalize and can add an
escalation (ADR-001 §3). In live Gemini runs it escalated tickets that code had
already decided:

- An eligible refund that was already pending approval went to tier2 with
  only `approval_required`. The approval queue is already the human step for
  that refund, so a second human queue doubled the work. (PR #20)
- Injection-only tickets (inject-002, inject-007) went to the security queue,
  though `docs/contracts/behavior.md` says to refuse them without escalating.
  (PR #21)
- A refund blocked by policy (outside the window, toolfail-009) went to tier2
  with `approval_required`, though nothing was left to approve. (PR #21)

## Decision

`triagewright.agent.coordinator._left_to_code` decides when a reviewer
escalation adds nothing. `Coordinator._route` ignores the review when every
reason it gives is one code already handled, and triage found no human need
that code left unhandled (including `needs_human` with no reason). Handled
reasons:

- `approval_required`, once the request's one action is decided: pending
  approval (the approval queue is that human step) or blocked (nothing to
  approve).
- `injection_suspected`, when code flagged injection before the reviewer's own
  screen (`RequestRun.flagged_before_review`) and the draft refuses. Code's
  injection rule in `_route` still sends any write or handoff request to the
  security queue.

The reviewer can still add an escalation for any other reason, and it can never
remove one that code makes. The triage and reviewer prompts say a refund
outcome alone, or an injection-only ticket whose reply refuses it, needs no
escalation.

## Consequences

- When the review's only reason is `approval_required`, the request has one
  action, and triage found no unhandled human need, a pending refund has one
  human owner, its approver, and a refund blocked by policy follows the refund
  table, not the reviewer. An escalating review still counts when it gives no
  reasons or a reason code didn't handle, or triage found an unhandled human
  need.
- An injection-only ticket ends as `refused`, as the behavior contract says,
  when code flagged injection before the review, the draft refuses, every
  review reason is handled, and triage found no unhandled need. Otherwise the
  reviewer's injection escalation still counts. inject-002 came back `refused` live
  before the daily free quota ran out; the other affected cases wait for the
  next full live run.
- The rule is narrow on purpose: every reason handled, no unhandled triage
  need, and, for `approval_required`, exactly one action.
- Live on 10 `happy-*` cases after PR #20: 90% pass (was 70%), 0 unsafe.

## Alternatives

- **Always honor the reviewer.** Rejected: it double-queued pending refunds and
  escalated tickets the contract says to refuse.
- **Drop reasons one by one and keep the rest.** Rejected: a review whose
  remaining reasons are thin would still escalate on reasons code had handled,
  and the handoff summary would explain the wrong thing.
- **Remove the reviewer.** Rejected: it catches human needs the routing rules
  don't encode, such as an ungrounded or low-confidence draft, and its screen
  is the request's last.
- **Prompt-only fix.** Not enough on its own: the prompts were updated, but code
  enforces the rule.
