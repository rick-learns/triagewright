"""Tool calls for one request: the retrying gateway wrapper, result screening, and citations."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence

from pydantic import ValidationError

from triagewright.agent.common import Fallback, LocalTool, as_data
from triagewright.agent.state import RunState
from triagewright.contracts.agents import LoopStop, ProposedAction
from triagewright.contracts.common import JsonObject, canonical_json
from triagewright.contracts.guardrails import ContentSource
from triagewright.contracts.llm import ToolDefinition
from triagewright.contracts.rag import Citation
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CallContext,
    Caller,
    KBHit,
    KBSearchResult,
    ListOrdersInput,
    OrderList,
    SearchKBInput,
    ToolName,
    ToolResult,
)
from triagewright.contracts.world import Customer, EscalationReason

_HANDOFF_FLOOR_S = 1.0


def _search_key(arguments: JsonObject) -> str | None:
    """``search_kb`` arguments with defaults filled in, so equivalent searches compare equal.
    The gateway reads no categories, or an empty list, as unfiltered, and their order as a set."""
    try:
        search = SearchKBInput.model_validate(arguments)
    except ValidationError:
        return None
    categories = sorted(set(search.categories)) if search.categories else None
    return canonical_json(
        search.model_copy(update={"categories": categories}).model_dump(mode="json")
    )


class ToolCalls(RunState):
    """The request's tool calls, on top of its state."""

    def context(self, caller: Caller) -> CallContext:
        return CallContext(
            request_id=self.request_id,
            ticket_id=self.ticket.ticket_id,
            customer_id=self.ticket.customer_id,
            caller=caller,
        )

    async def call_tool(
        self, name: ToolName, arguments: JsonObject, caller: Caller, *, handoff: bool = False
    ) -> ToolResult:
        """Call the gateway, retrying retryable errors up to ``tool_retries`` times.

        Every attempt is a raw gateway call, so ``ExpectedOutcome.attempts`` counts them all.
        A retryable error that outlasts the retries is recorded as a tool failure.
        """
        ctx = self.context(caller)
        attempt = 0
        while True:
            attempt += 1
            left = self.remaining(handoff=handoff)
            if left <= 0:
                if not (handoff and attempt == 1):
                    raise Fallback(
                        LoopStop.TIMEOUT, EscalationReason.LOOP_LIMIT, "time budget used up"
                    )
                left = _HANDOFF_FLOOR_S  # one handoff attempt even if work overran its budget
            attrs = {"tool": name.value, "caller": caller, "attempt": attempt}
            with self.telemetry.span(f"tool.{name.value}", SpanKind.TOOL_CALL, attrs) as span:
                started = self.clock()
                try:
                    async with asyncio.timeout(left):
                        result = await self.gateway.call(name, arguments, ctx)
                except TimeoutError:
                    span.set_error("timeout")
                    raise Fallback(
                        LoopStop.TIMEOUT, EscalationReason.LOOP_LIMIT, f"{name.value} timed out"
                    ) from None
                self.telemetry.record_tool_call(result, (self.clock() - started) * 1000.0)
                if result.error is not None:
                    span.set_error(result.error.category.value)
            error = result.error
            if error is None or not error.retryable:
                return result
            if attempt > self.config.tool_retries:
                break
            if error.retry_after_ms is not None:
                delay = error.retry_after_ms / 1000.0
            else:
                delay = self.config.retry_backoff_s * 2 ** (attempt - 1)
            if delay >= self.remaining(handoff=handoff):
                break
            if delay > 0:
                await self.sleep(delay)
        self.tool_failures.append(name)
        return result

    def observe(self, result: ToolResult, arguments: JsonObject) -> str:
        """Screen a tool result, keep its facts and KB hits, and return what models see."""
        content = canonical_json(result.model_dump(mode="json"))
        hits: list[KBHit] | None = None
        if result.tool is ToolName.SEARCH_KB and result.data is not None:
            try:
                hits = KBSearchResult.model_validate(result.data).hits
            except ValidationError:
                hits = None
        if hits is not None:
            flagged = False
            for hit in hits:
                self.kb_hits.setdefault(hit.chunk_id, hit)
                if self.screen(f"{hit.title}\n{hit.text}", ContentSource.KB_DOCUMENT):
                    flagged = True
                    self.planted_in.append(hit.chunk_id)
            return as_data(content, flagged)
        # Planted in a record or page, not the customer's ask: an attack on the system.
        flagged = self.screen(content, ContentSource.TOOL_OUTPUT)
        if flagged:
            self.planted_in.append(result.tool.value)
        if result.tool is ToolName.GET_CUSTOMER and result.data is not None:
            with contextlib.suppress(ValidationError):
                self.customer_status = Customer.model_validate(result.data).status
        if result.data is not None:
            fact = f"{result.tool.value} {canonical_json(arguments)}: {canonical_json(result.data)}"
            self.facts.append(as_data(fact, flagged))
        return as_data(content, flagged)

    def _list_orders(self, query: ListOrdersInput, data: JsonObject) -> None:
        self.listed_orders = OrderList.model_validate(data).orders
        self.listed_all = len(self.listed_orders) < query.limit

    def business_tool(self, name: ToolName, caller: Caller) -> LocalTool:
        spec = TOOL_SPECS[name]

        async def handler(arguments: JsonObject) -> tuple[str, bool]:
            if name is ToolName.SEARCH_KB and (key := _search_key(arguments)) is not None:
                self.kb_searches.add(key)
            result = await self.call_tool(name, arguments, caller)
            if name is ToolName.LIST_ORDERS and result.data is not None:
                with contextlib.suppress(ValidationError):
                    self._list_orders(ListOrdersInput.model_validate(arguments), result.data)
            return self.observe(result, arguments), result.is_error

        definition = ToolDefinition(
            name=name.value, description=spec.description, input_schema=spec.input_schema()
        )
        return LocalTool(definition=definition, handler=handler)

    def searched(self, arguments: JsonObject) -> bool:
        """Whether this request already ran an equivalent ``search_kb`` call."""
        return _search_key(arguments) in self.kb_searches

    def cite(self, chunk_id: str, quote: str) -> Citation | None:
        """A citation only for a trusted hit seen in this request whose text holds the quote."""
        hit = self.kb_hits.get(chunk_id)
        quote = quote.strip()
        if hit is None or not hit.trusted or not quote or quote not in hit.text:
            return None
        return Citation(doc_id=hit.doc_id, chunk_id=hit.chunk_id, title=hit.title, quote=quote)

    def cite_all(self, candidates: Sequence[tuple[str, str]]) -> list[Citation]:
        seen: set[tuple[str, str]] = set()
        citations: list[Citation] = []
        for chunk_id, quote in candidates:
            citation = self.cite(chunk_id, quote)
            if citation is None or (citation.chunk_id, citation.quote) in seen:
                continue
            seen.add((citation.chunk_id, citation.quote))
            citations.append(citation)
        return citations

    def defer(self, action: ProposedAction) -> None:
        """Queue an irreversible proposal until after the request's last ``screen``."""
        self.irreversible_requested = True
        key = (action.tool, canonical_json(action.arguments))
        if all((a.tool, canonical_json(a.arguments)) != key for a in self.deferred):
            self.deferred.append(action)
