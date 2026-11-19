"""In-memory approval store implementing the ``ApprovalStore`` port."""

from __future__ import annotations

from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
)


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self.items: dict[str, ApprovalRequest] = {}

    def submit(self, request: ApprovalRequest) -> ApprovalRequest:
        return self.items.setdefault(request.approval_id, request)

    def get(self, approval_id: str) -> ApprovalRequest | None:
        return self.items.get(approval_id)

    def decide(self, decision: ApprovalDecision) -> ApprovalRequest:
        current = self.items[decision.approval_id]
        if current.status is not ApprovalStatus.PENDING:
            raise ValueError(f"{decision.approval_id} is already {current.status.value}")
        status = ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED
        updated = current.model_copy(update={"status": status})
        self.items[decision.approval_id] = updated
        return updated

    def mark_executed(self, approval_id: str) -> ApprovalRequest:
        current = self.items[approval_id]
        if current.status is not ApprovalStatus.APPROVED:
            raise ValueError(f"{approval_id} is {current.status.value}, not approved")
        updated = current.model_copy(update={"status": ApprovalStatus.EXECUTED})
        self.items[approval_id] = updated
        return updated

    def list_pending(self) -> list[ApprovalRequest]:
        return [a for a in self.items.values() if a.status is ApprovalStatus.PENDING]
