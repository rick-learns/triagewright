"""Reference ``ActionExecutor`` over the fake gateway, guardrails, and approval store."""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError

from triagewright.contracts.agents import (
    ActionOutcome,
    ActionStatus,
    ProposedAction,
)
from triagewright.contracts.common import JsonObject, canonical_json, stable_id
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
)
from triagewright.contracts.policy import (
    approval_token,
    refund_idempotency_key,
)
from triagewright.contracts.ports import Guardrails
from triagewright.contracts.tools import (
    CODE_OWNED_TOOLS,
    IRREVERSIBLE_TOOLS,
    CallContext,
    ErrorCategory,
    RefundEligibility,
    RefundProposal,
    ToolName,
    ToolResult,
    proposal_model,
)
from triagewright.testing.fakes.approvals import InMemoryApprovalStore
from triagewright.testing.fakes.gateway import TEST_APPROVAL_KEY, FakeToolGateway


class FakeActionExecutor:
    """Reference implementation of the ``ActionExecutor`` port over the fakes."""

    def __init__(
        self,
        gateway: FakeToolGateway,
        guardrails: Guardrails,
        approvals: InMemoryApprovalStore,
        *,
        now: datetime,
        approval_key: bytes = TEST_APPROVAL_KEY,
        tool_retries: int = 2,
    ) -> None:
        self.gateway = gateway
        self.guardrails = guardrails
        self.approvals = approvals
        self.now = now
        self._key = approval_key
        self._retries = tool_retries

    async def _call_with_retries(
        self, name: ToolName, arguments: JsonObject, ctx: CallContext
    ) -> tuple[ToolResult, int]:
        attempts = 0
        while True:
            attempts += 1
            result = await self.gateway.call(name, arguments, ctx)
            retryable = result.error is not None and result.error.retryable
            if not retryable or attempts > self._retries:
                return result, attempts

    async def propose(
        self, action: ProposedAction, ctx: CallContext, *, injection_suspected: bool
    ) -> ActionOutcome:
        if action.tool in CODE_OWNED_TOOLS:
            verdict = GuardrailVerdict(
                decision=GuardrailDecision.BLOCK, rule_ids=["tool.code_owned"]
            )
            return ActionOutcome(action=action, status=ActionStatus.BLOCKED, verdict=verdict)
        try:
            args = proposal_model(action.tool).model_validate(action.arguments)
        except ValidationError as exc:
            result = ToolResult.fail(
                action.tool, ErrorCategory.VALIDATION, str(exc.errors()[0]["msg"])
            )
            return ActionOutcome(action=action, status=ActionStatus.BLOCKED, result=result)
        arguments: JsonObject = args.model_dump(mode="json")
        approval_id = stable_id("apr", ctx.ticket_id, action.tool, canonical_json(arguments))
        if action.tool in IRREVERSIBLE_TOOLS:
            if injection_suspected:
                verdict = GuardrailVerdict(
                    decision=GuardrailDecision.BLOCK,
                    rule_ids=["injection.write_blocked"],
                    injection_suspected=True,
                )
                return ActionOutcome(action=action, status=ActionStatus.BLOCKED, verdict=verdict)
            existing = self.approvals.get(approval_id)
            if existing is not None and existing.customer_id == ctx.customer_id:
                return ActionOutcome(
                    action=action,
                    status=_STORED_STATUS[existing.status],
                    verdict=existing.verdict,
                    approval_id=approval_id,
                )
        elig = None
        if isinstance(args, RefundProposal):
            check, attempts = await self._call_with_retries(
                ToolName.CHECK_REFUND_ELIGIBILITY,
                {"order_id": args.order_id, "amount_cents": args.amount_cents},
                ctx,
            )
            if check.is_error or check.data is None:
                return ActionOutcome(
                    action=action, status=ActionStatus.FAILED, result=check, attempts=attempts
                )
            elig = RefundEligibility.model_validate(check.data)
        policy_ctx = PolicyContext(
            ticket_id=ctx.ticket_id,
            customer_id=ctx.customer_id,
            eligibility=elig,
            injection_suspected=injection_suspected,
            now=self.now,
        )
        verdict = self.guardrails.check_action(action, policy_ctx)
        blocked = ActionOutcome(
            action=action, status=ActionStatus.BLOCKED, verdict=verdict, eligibility=elig
        )
        if verdict.decision is GuardrailDecision.BLOCK:
            return blocked
        if verdict.decision is GuardrailDecision.REQUIRE_APPROVAL:
            if action.tool not in IRREVERSIBLE_TOOLS or ctx.customer_id is None:
                return blocked
            stored = self.approvals.submit(
                ApprovalRequest(
                    approval_id=approval_id,
                    ticket_id=ctx.ticket_id,
                    customer_id=ctx.customer_id,
                    tool=action.tool,
                    arguments=arguments,
                    idempotency_key=refund_idempotency_key(approval_id),
                    rationale=action.rationale,
                    verdict=verdict,
                    requested_at=self.now,
                )
            )
            return ActionOutcome(
                action=action,
                status=_STORED_STATUS[stored.status],
                verdict=verdict,
                approval_id=approval_id,
                eligibility=elig,
            )
        if action.tool in IRREVERSIBLE_TOOLS:  # a guardrail bug must not run a refund
            return blocked
        result = await self.gateway.call(action.tool, arguments, ctx)
        status = ActionStatus.FAILED if result.is_error else ActionStatus.EXECUTED
        return ActionOutcome(
            action=action, status=status, result=result, verdict=verdict, attempts=1
        )

    async def decide(self, decision: ApprovalDecision) -> ActionOutcome:
        current = self.approvals.get(decision.approval_id)
        if current is not None and current.status is ApprovalStatus.APPROVED and decision.approved:
            return await self.execute_approved(current.approval_id)  # resume after a failure
        request = self.approvals.decide(decision)  # raises on any other repeat decision
        if request.status is ApprovalStatus.APPROVED:
            return await self.execute_approved(request.approval_id)
        return ActionOutcome(
            action=_stored_action(request),
            status=ActionStatus.SKIPPED,
            approval_id=request.approval_id,
        )

    async def execute_approved(self, approval_id: str) -> ActionOutcome:
        request = self.approvals.get(approval_id)
        if request is None:
            raise KeyError(approval_id)
        action = _stored_action(request)
        if request.status is not ApprovalStatus.APPROVED:
            return ActionOutcome(
                action=action, status=ActionStatus.BLOCKED, approval_id=approval_id
            )
        proposal = RefundProposal.model_validate(request.arguments)
        token = approval_token(
            self._key,
            approval_id=approval_id,
            order_id=proposal.order_id,
            amount_cents=proposal.amount_cents,
            idempotency_key=request.idempotency_key,
        )
        ctx = CallContext(
            request_id=stable_id("req", request.ticket_id, "approval", approval_id),
            ticket_id=request.ticket_id,
            customer_id=request.customer_id,
            caller="human",
        )
        arguments: JsonObject = {
            **request.arguments,
            "idempotency_key": request.idempotency_key,
            "approval_id": approval_id,
            "approval_token": token,
        }
        result = await self.gateway.call(request.tool, arguments, ctx)
        if result.is_error:
            return ActionOutcome(
                action=action,
                status=ActionStatus.FAILED,
                result=result,
                approval_id=approval_id,
                attempts=1,
            )
        self.approvals.mark_executed(approval_id)
        return ActionOutcome(
            action=action,
            status=ActionStatus.EXECUTED,
            result=result,
            approval_id=approval_id,
            attempts=1,
        )


# What re-proposing an existing approval reports. Never "executed" in-request.
_STORED_STATUS: dict[ApprovalStatus, ActionStatus] = {
    ApprovalStatus.PENDING: ActionStatus.PENDING_APPROVAL,
    ApprovalStatus.APPROVED: ActionStatus.PENDING_APPROVAL,
    ApprovalStatus.REJECTED: ActionStatus.BLOCKED,
    ApprovalStatus.EXECUTED: ActionStatus.SKIPPED,
}


def _stored_action(request: ApprovalRequest) -> ProposedAction:
    return ProposedAction(
        tool=request.tool, arguments=request.arguments, rationale=request.rationale[:500]
    )
