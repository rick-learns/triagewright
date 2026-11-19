"""Minimal guardrails: no redaction or screening, a basic action policy (lane D replaces it)."""

from __future__ import annotations

from triagewright.contracts.agents import (
    ProposedAction,
)
from triagewright.contracts.guardrails import (
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
    RedactionResult,
)
from triagewright.contracts.tools import (
    IRREVERSIBLE_TOOLS,
    ToolName,
)


class PassthroughGuardrails:
    """No redaction, no screening, and a minimal action policy (lane D replaces it).

    Blocks writes once injection is suspected and refunds without a passing
    eligibility check. Every other irreversible action needs approval.
    """

    def redact(self, text: str) -> RedactionResult:
        return RedactionResult(text=text)

    def screen(self, text: str, source: ContentSource) -> GuardrailVerdict:
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)

    def check_action(self, action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict:
        if ctx.injection_suspected and action.tool in IRREVERSIBLE_TOOLS:
            return GuardrailVerdict(
                decision=GuardrailDecision.BLOCK,
                rule_ids=["injection.write_blocked"],
                injection_suspected=True,
            )
        if action.tool is ToolName.ISSUE_REFUND:
            elig = ctx.eligibility
            if elig is None or not elig.eligible:
                reasons = [r.value for r in elig.reasons] if elig else ["no eligibility check"]
                return GuardrailVerdict(
                    decision=GuardrailDecision.BLOCK,
                    rule_ids=[f"refund.{r}" for r in reasons],
                    reasons=reasons,
                )
        if action.tool in IRREVERSIBLE_TOOLS:
            return GuardrailVerdict(
                decision=GuardrailDecision.REQUIRE_APPROVAL, rule_ids=["approval.irreversible"]
            )
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)
