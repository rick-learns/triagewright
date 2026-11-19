"""Per-request state, the bounded model loop, and the retrying tool-call wrapper.

One :class:`RequestRun` exists per ``resolve`` call, so a coordinator can serve
requests concurrently. It holds everything code owns for the request: the
redacted ticket, the ``injection_suspected`` flag (the OR of every ``screen``),
verified facts, the knowledge-base hits citations are checked against, the
deferred irreversible proposals, the executor outcomes, and the time budget.

The pieces live in ``common`` (signals, local tools, the data wrapper), ``state``
(fields, time, screens), ``tool_calls`` (gateway calls, observation, citations), and
``loop`` (model calls and ``run_loop``); :class:`RequestRun` stacks them.
"""

from __future__ import annotations

from triagewright.agent.common import Clock as Clock
from triagewright.agent.common import Fallback as Fallback
from triagewright.agent.common import Handler as Handler
from triagewright.agent.common import LocalTool as LocalTool
from triagewright.agent.common import Refused as Refused
from triagewright.agent.common import Sleep as Sleep
from triagewright.agent.common import as_data as as_data
from triagewright.agent.common import escape_data as escape_data
from triagewright.agent.common import validation_message as validation_message
from triagewright.agent.loop import ModelLoop
from triagewright.agent.loop import _append as _append
from triagewright.agent.tool_calls import _HANDOFF_FLOOR_S as _HANDOFF_FLOOR_S
from triagewright.agent.tool_calls import _search_key as _search_key


class RequestRun(ModelLoop):
    """One request's state, tool calls, and model loop."""
