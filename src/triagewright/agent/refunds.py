"""Refund rules: the finalize gate, the replies code writes, and settling with the policy
citation."""

from __future__ import annotations

import re
import sys

from triagewright.agent import prompts
from triagewright.agent.finalize import CitationCandidate, FinalizeArgs
from triagewright.agent.runtime import RequestRun
from triagewright.contracts.agents import EscalationDecision, TicketCategory
from triagewright.contracts.common import JsonObject
from triagewright.contracts.rag import Citation
from triagewright.contracts.tools import ToolName
from triagewright.contracts.world import EscalationQueue

# An order id anywhere in the ticket (``contracts.common.OrderId``).
_ORDER_ID = re.compile(r"\bord_[a-z0-9]{4,32}\b")

_REFUND_POLICY_DOC = "kb_refund_policy"
_POLICY_SEARCH: JsonObject = {"query": "refund policy"}


class RefundRules:
    """The coordinator's refund-reply methods."""

    def _check_final(self, run: RequestRun, final: FinalizeArgs) -> str | None:
        """A refund reply needs a verified refund-policy citation and a refund proposal, so
        the executor decides the outcome (behavior.md). Code checks both: a model can skip
        a prompt rule. Without a proposal, only the replies code writes are accepted (see
        ``_own_refund_reply``)."""
        if not (_refund_requested(run) or _refund_proposed(run)):
            return None
        citations = run.cite_all([(c.chunk_id, c.quote) for c in final.citations])
        cited = any(c.doc_id == _REFUND_POLICY_DOC for c in citations)
        missing = [
            message
            for message, ok in (
                (prompts.PROPOSE_REFUND_FIRST, _refund_decidable(run, final.status)),
                (prompts.CITE_REFUND_POLICY, cited),
            )
            if not ok
        ]
        return " ".join(missing) or None

    def _own_refund_reply(self, run: RequestRun, final: FinalizeArgs) -> FinalizeArgs:
        """A refund reply with no proposal behind it is written by code, never the model,
        so no model can deny or settle a refund: a sign-in request for an anonymous ticket,
        or a question about which order when the records leave it open. A reply beside a
        refund proposal is code's too, before the reviewer sees it: the model writes before
        the executor decides (``_settled`` normally writes it; this covers any other path)."""
        if _refund_proposed(run):
            return final.model_copy(update={"reply": prompts.REFUND_OUTCOME_REPLY})
        if not _refund_requested(run):
            return final
        if run.ticket.customer_id is None:
            reply = prompts.SIGN_IN_FOR_REFUND
        elif _order_unclear(run):
            reply = prompts.WHICH_ORDER_FOR_REFUND
        else:
            # ``_check_final`` accepts this proposal-free reply only with injection suspected,
            # and then security takes the ticket and ``_refund_reply`` drops the model's text.
            return final
        return final.model_copy(update={"status": "needs_info", "reply": reply})

    @staticmethod
    def _refund_reply(reply: str, decision: EscalationDecision | None) -> str:
        """Code states every refund outcome (``_note``), and a security handoff never carries
        model text. What's left is a reply code wrote: ``_own_refund_reply`` replaced any
        model reply beside a proposal, and a refusal's reply is code's."""
        security = decision is not None and decision.queue is EscalationQueue.SECURITY
        if security:
            return prompts.FALLBACK_REPLY
        return reply


def _refund_requested(run: RequestRun) -> bool:
    """Triage's category, or its flag for a refund asked for under another category."""
    assert run.triage is not None
    return run.triage.category is TicketCategory.REFUND_REQUEST or run.triage.refund_requested


async def _settled(run: RequestRun) -> FinalizeArgs | None:
    """A ticket is settled once a refund is proposed and the policy is cited: the executor
    decides the outcome and code states it (``_note``), so a model's reply would only guess
    at it. Code writes the reply and the citation instead, searching for the policy itself
    once the turn's own tools are done if no model did."""
    if not _refund_proposed(run):
        return None
    if _policy_citation(run) is None and not run.searched(_POLICY_SEARCH):
        # Once a request: a failed or empty lookup isn't repeated, nor is a model's own.
        search = run.business_tool(ToolName.SEARCH_KB, "coordinator")
        await search.handler(_POLICY_SEARCH)
    citation = _policy_citation(run)
    if citation is None:
        return None
    cited = CitationCandidate(chunk_id=citation.chunk_id, quote=citation.quote)
    return FinalizeArgs(status="resolved", reply=prompts.REFUND_OUTCOME_REPLY, citations=[cited])


def _policy_citation(run: RequestRun) -> Citation | None:
    """A trusted refund-policy hit, quoting its first sentence. Hits go in page order, not
    rank, so the policy's opening rule (the 30-day window every refund is decided by) is
    cited when it was retrieved, rather than whichever section ranked first, such as what
    can't be refunded beside an eligible refund."""
    hits = [hit for hit in run.kb_hits.values() if hit.doc_id == _REFUND_POLICY_DOC]
    for hit in sorted(hits, key=lambda hit: _ordinal(hit.chunk_id)):
        if (citation := run.cite(hit.chunk_id, first_sentence(hit.text))) is not None:
            return citation
    return None


def _ordinal(chunk_id: str) -> int:
    """A chunk's place in its page (``<doc_id>#<ordinal>``); unnumbered chunks go last."""
    ordinal = chunk_id.rpartition("#")[2]
    return int(ordinal) if ordinal.isdigit() else sys.maxsize


def first_sentence(text: str) -> str:
    """The first sentence of a chunk's first line of prose (chunks open with a heading), or
    its heading when it has no prose."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prose = [line for line in lines if not line.startswith("#")] or [
        line.lstrip("# ") for line in lines
    ]
    return prose[0].split(". ")[0][:400] if prose else ""


def _refund_proposed(run: RequestRun) -> bool:
    proposals = [*run.deferred, *(o.action for o in run.outcomes)]
    return any(p.tool is ToolName.ISSUE_REFUND for p in proposals)


def _order_unclear(run: RequestRun) -> bool:
    """The ticket names no order id, and the customer's orders (from ``list_orders``)
    don't narrow to one: by a product the ticket names, else by count. A full page
    may have more orders behind it, so it never narrows to one."""
    ticket = f"{run.ticket.subject}\n{run.ticket.body}"
    if run.listed_orders is None or _ORDER_ID.search(ticket) is not None:
        return False
    if not run.listed_all:
        return True  # older orders may match too
    text = ticket.lower()
    named = [o for o in run.listed_orders if any(i.name.lower() in text for i in o.items)]
    return len(named or run.listed_orders) != 1


def _refund_decidable(run: RequestRun, status: str) -> bool:
    """Whether a refund finalize leaves the outcome to code, not the model.

    The executor decides from a queued ``issue_refund`` proposal (with injection suspected,
    it blocks the proposal and the ticket goes to security). Without one, only a
    ``needs_info`` that code rewrites is accepted: no signed-in customer, or no clear order.
    With injection suspected it needs none: ``_decide`` sends the ask to security.
    """
    if _refund_proposed(run) or run.ticket.customer_id is None or run.injection:
        return True
    return status == "needs_info" and _order_unclear(run)
