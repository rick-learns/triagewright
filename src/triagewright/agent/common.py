"""Building blocks of the runtime: control-flow signals, local tools, and the data wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import ValidationError

from triagewright.agent import prompts
from triagewright.contracts.agents import LoopStop
from triagewright.contracts.common import JsonObject
from triagewright.contracts.llm import ToolDefinition
from triagewright.contracts.world import EscalationReason

type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]
type Handler = Callable[[JsonObject], Awaitable[tuple[str, bool]]]


class Fallback(Exception):  # noqa: N818 - a control-flow signal, not an error type
    """A bound was hit or a model failed for good: hand the ticket to a human."""

    def __init__(self, stop: LoopStop, reason: EscalationReason, detail: str) -> None:
        super().__init__(detail)
        self.stop = stop
        self.reason = reason
        self.detail = detail


class Refused(Exception):  # noqa: N818 - a control-flow signal, not an error type
    """A model returned ``stop_reason=refusal``."""


@dataclass(frozen=True)
class LocalTool:
    definition: ToolDefinition
    handler: Handler


def validation_message(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()]
    return ("Invalid arguments. " + "; ".join(parts))[:1000]


def as_data(content: str, flagged: bool) -> str:
    """Wrap flagged content in delimiters with a "data, not instructions" note."""
    if not flagged:
        return content
    return f"<data>{escape_data(content)}</data>\n{prompts.FLAGGED_NOTE}"


def escape_data(content: str) -> str:
    return content.replace("</data>", "<\\/data>")
