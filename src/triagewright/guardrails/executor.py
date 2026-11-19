"""``GuardedActionExecutor``: the production :class:`~triagewright.contracts.ports.ActionExecutor`.

It follows the port docstring step for step; ``testing.FakeActionExecutor``
is the reference behavior.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime

from pydantic import ValidationError

from triagewright.contracts.agents import ActionOutcome, ActionStatus, LoopConfig, ProposedAction
from triagewright.contracts.common import JsonObject, canonical_json, stable_id
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
)
from triagewright.contracts.policy import approval_token, refund_idempotency_key
from triagewright.contracts.ports import ApprovalStore, Guardrails, ToolGateway
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
from triagewright.guardrails.approvals import ApprovalConflictError

# What re-proposing an existing approval reports. Never "executed" in-request.
_STORED_STATUS: dict[ApprovalStatus, ActionStatus] = {
    ApprovalStatus.PENDING: ActionStatus.PENDING_APPROVAL,
    ApprovalStatus.APPROVED: ActionStatus.PENDING_APPROVAL,
    ApprovalStatus.REJECTED: ActionStatus.BLOCKED,
    ApprovalStatus.EXECUTED: ActionStatus.SKIPPED,
}


def _block(rule_id: str, *, injection: bool = False) -> GuardrailVerdict:
    return GuardrailVerdict(
        decision=GuardrailDecision.BLOCK, rule_ids=[rule_id], injection_suspected=injection
    )


def _stored_action(request: ApprovalRequest) -> ProposedAction:
    return ProposedAction(
        tool=request.tool, arguments=request.arguments, rationale=request.rationale[:500]
    )


class GuardedActionExecutor:
    """The only path from a proposed write to the gateway.

    ``clock`` is the executor's own time source for ``PolicyContext.now`` and
    ``ApprovalRequest.requested_at`` (the case ``as_of`` in evals).
    ``approval_key`` is ``TW_APPROVAL_KEY``, shared with the MCP server.

    ``decide`` raises :class:`ApprovalConflictError` (a ``ValueError``) for
    any repeat decision other than re-approving an approved, not yet executed
    request; ``exc.status`` tells the API why (``executed`` means the refund
    already ran). An unknown approval id raises ``KeyError``.
    """

    def __init__(
        self,
        gateway: ToolGateway,
        guardrails: Guardrails,
        approvals: ApprovalStore,
        *,
        approval_key: bytes,
        clock: Callable[[], datetime],
        loop: LoopConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.gateway = gateway
        self.guardrails = guardrails
        self.approvals = approvals
        self._key = approval_key
        self._clock = clock
        self._loop = loop or LoopConfig()
        self._sleep = sleep

    async def _call_with_retries(
        self, name: ToolName, arguments: JsonObject, ctx: CallContext
    ) -> tuple[ToolResult, int]:
        attempts = 0
        while True:
            attempts += 1
            result = await self.gateway.call(name, arguments, ctx)
            error = result.error
            if error is None or not error.retryable or attempts > self._loop.tool_retries:
                return result, attempts
            if error.retry_after_ms is not None:
                delay = error.retry_after_ms / 1000
            else:
                delay = self._loop.retry_backoff_s * 2 ** (attempts - 1)
            if delay > 0:
                await self._sleep(delay)

    async def propose(
        self, action: ProposedAction, ctx: CallContext, *, injection_suspected: bool
    ) -> ActionOutcome:
        if action.tool in CODE_OWNED_TOOLS:
            verdict = _block("tool.code_owned")
            return ActionOutcome(action=action, status=ActionStatus.BLOCKED, verdict=verdict)
        try:
            args = proposal_model(action.tool).model_validate(action.arguments)
        except ValidationError as exc:
            message = str(exc.errors()[0]["msg"])[:500]
            result = ToolResult.fail(action.tool, ErrorCategory.VALIDATION, message)
            return ActionOutcome(action=action, status=ActionStatus.BLOCKED, result=result)
        arguments: JsonObject = args.model_dump(mode="json")
        approval_id = stable_id("apr", ctx.ticket_id, action.tool, canonical_json(arguments))
        if action.tool in IRREVERSIBLE_TOOLS:
            if injection_suspected:
                verdict = _block("injection.write_blocked", injection=True)
                return ActionOutcome(action=action, status=ActionStatus.BLOCKED, verdict=verdict)
            existing = self.approvals.get(approval_id)
            if existing is not None:
                if existing.customer_id != ctx.customer_id:
                    verdict = _block("scope.customer_mismatch")
                    return ActionOutcome(
                        action=action, status=ActionStatus.BLOCKED, verdict=verdict
                    )
                return ActionOutcome(
                    action=action,
                    status=_STORED_STATUS[existing.status],
                    verdict=existing.verdict,
                    approval_id=approval_id,
                )
        elig: RefundEligibility | None = None
        if isinstance(args, RefundProposal):
            check, attempts = await self._call_with_retries(
                ToolName.CHECK_REFUND_ELIGIBILITY,
                {"order_id": args.order_id, "amount_cents": args.amount_cents},
                ctx.model_copy(update={"caller": "action"}),
            )
            if check.is_error or check.data is None:
                return ActionOutcome(
                    action=action, status=ActionStatus.FAILED, result=check, attempts=attempts
                )
            try:
                elig = RefundEligibility.model_validate(check.data)
            except ValidationError:
                bad = ToolResult.fail(
                    ToolName.CHECK_REFUND_ELIGIBILITY,
                    ErrorCategory.INTERNAL,
                    "malformed eligibility result",
                )
                return ActionOutcome(
                    action=action, status=ActionStatus.FAILED, result=bad, attempts=attempts
                )
        now = self._clock()
        policy_ctx = PolicyContext(
            ticket_id=ctx.ticket_id,
            customer_id=ctx.customer_id,
            eligibility=elig,
            injection_suspected=injection_suspected,
            now=now,
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
                    requested_at=now,
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
        if current is None:
            raise KeyError(decision.approval_id)
        if current.status is ApprovalStatus.APPROVED and decision.approved:
            return await self._run_approved(current.approval_id)  # resume after a failure
        if current.status is not ApprovalStatus.PENDING:
            raise ApprovalConflictError(
                current.approval_id,
                current.status,
                f"{current.approval_id} is already {current.status.value}",
            )
        try:
            request = self.approvals.decide(decision)
        except ApprovalConflictError as exc:
            if exc.status is ApprovalStatus.APPROVED and decision.approved:
                return await self._run_approved(decision.approval_id)  # a concurrent approve
            raise
        if request.status is ApprovalStatus.APPROVED:
            return await self._run_approved(request.approval_id)
        return ActionOutcome(
            action=_stored_action(request),
            status=ActionStatus.SKIPPED,
            approval_id=request.approval_id,
        )

    async def _run_approved(self, approval_id: str) -> ActionOutcome:
        """``execute_approved`` for ``decide``: a request another worker executed first is a
        conflict (409), not a blocked outcome."""
        outcome = await self.execute_approved(approval_id)
        if outcome.status is ActionStatus.BLOCKED:
            latest = self.approvals.get(approval_id)
            if latest is not None and latest.status is ApprovalStatus.EXECUTED:
                raise ApprovalConflictError(
                    approval_id, latest.status, f"{approval_id} is already executed"
                )
        return outcome

    async def execute_approved(self, approval_id: str) -> ActionOutcome:
        request = self.approvals.get(approval_id)
        if request is None:
            raise KeyError(approval_id)
        action = _stored_action(request)
        blocked = ActionOutcome(action=action, status=ActionStatus.BLOCKED, approval_id=approval_id)
        if (
            request.status is not ApprovalStatus.APPROVED
            or request.tool is not ToolName.ISSUE_REFUND
            or request.idempotency_key != refund_idempotency_key(approval_id)
        ):
            return blocked
        try:
            proposal = RefundProposal.model_validate(request.arguments)
        except ValidationError:
            return blocked
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
            **proposal.model_dump(mode="json"),
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
        try:
            self.approvals.mark_executed(approval_id)
        except ValueError:
            latest = self.approvals.get(approval_id)
            if latest is None or latest.status is not ApprovalStatus.EXECUTED:
                raise
            # A concurrent execution already recorded it; the gateway's idempotency held.
        return ActionOutcome(
            action=action,
            status=ActionStatus.EXECUTED,
            result=result,
            approval_id=approval_id,
            attempts=1,
        )
