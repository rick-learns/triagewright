"""``RuleGuardrails``: the production :class:`~triagewright.contracts.ports.Guardrails`."""

from __future__ import annotations

from pydantic import ValidationError

from triagewright.contracts.agents import ProposedAction
from triagewright.contracts.guardrails import (
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
    RedactionResult,
)
from triagewright.contracts.tools import (
    CODE_OWNED_TOOLS,
    IRREVERSIBLE_TOOLS,
    TOOL_SPECS,
    RefundProposal,
    SideEffect,
    ToolName,
    proposal_model,
)
from triagewright.guardrails import redaction, screening


def _verdict(
    decision: GuardrailDecision, rule_id: str, *reasons: str, injection: bool = False
) -> GuardrailVerdict:
    return GuardrailVerdict(
        decision=decision,
        rule_ids=[rule_id],
        reasons=list(reasons),
        injection_suspected=injection,
    )


def _block(rule_id: str, *reasons: str, injection: bool = False) -> GuardrailVerdict:
    return _verdict(GuardrailDecision.BLOCK, rule_id, *reasons, injection=injection)


class RuleGuardrails:
    """Pure, synchronous, deterministic guardrails.

    ``check_action`` decides, in this order (the first match wins):

    - ``tool.code_owned``: a model may never propose a code-owned tool.
    - ``action.invalid_arguments``: arguments must match ``proposal_model``.
    - reads are allowed.
    - ``injection.write_blocked``: no write once injection is suspected.
    - ``scope.no_customer``: an irreversible action needs a customer in scope.
    - refunds: ``refund.no_eligibility`` without a lookup;
      ``refund.eligibility_mismatch`` if the lookup is for another order;
      ``refund.<reason>`` for each reason ``policy.evaluate_refund``
      returned. That lookup is the only policy: the executor makes it for
      these exact arguments through the gateway, which applies the server's
      ``PolicyConfig``. Nothing here re-applies a limit, so the two can't
      disagree.
    - ``approval.irreversible``: every other irreversible action needs a human.
    """

    def redact(self, text: str) -> RedactionResult:
        return redaction.redact(text)

    def screen(self, text: str, source: ContentSource) -> GuardrailVerdict:
        return screening.screen(text, source)

    def check_action(self, action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict:
        tool = action.tool
        if tool in CODE_OWNED_TOOLS:
            return _block("tool.code_owned", f"{tool.value} is only called by coordinator code")
        try:
            args = proposal_model(tool).model_validate(action.arguments)
        except ValidationError as exc:
            return _block("action.invalid_arguments", str(exc.errors()[0]["msg"]))
        if TOOL_SPECS[tool].side_effect is SideEffect.READ:
            return GuardrailVerdict(decision=GuardrailDecision.ALLOW)
        if ctx.injection_suspected:
            return _block(
                "injection.write_blocked", "injection suspected in this request", injection=True
            )
        if tool in IRREVERSIBLE_TOOLS and ctx.customer_id is None:
            return _block("scope.no_customer", "no customer in scope")
        if tool is ToolName.ISSUE_REFUND:
            assert isinstance(args, RefundProposal)
            if (blocked := self._check_refund(args, ctx)) is not None:
                return blocked
        if tool in IRREVERSIBLE_TOOLS:
            return _verdict(
                GuardrailDecision.REQUIRE_APPROVAL,
                "approval.irreversible",
                f"{tool.value} is irreversible and needs a human approval",
            )
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)  # pragma: no cover - none yet

    def _check_refund(self, args: RefundProposal, ctx: PolicyContext) -> GuardrailVerdict | None:
        elig = ctx.eligibility
        if elig is None:
            return _block("refund.no_eligibility", "no eligibility check")
        if elig.order_id != args.order_id:
            return _block(
                "refund.eligibility_mismatch", "the eligibility check is for another order"
            )
        if not elig.eligible:
            reasons = [r.value for r in elig.reasons] or ["ineligible"]
            return GuardrailVerdict(
                decision=GuardrailDecision.BLOCK,
                rule_ids=[f"refund.{r}" for r in reasons],
                reasons=reasons,
            )
        return None
