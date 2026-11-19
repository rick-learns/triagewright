"""The Gemini and OpenRouter clients on one OpenAI-compatible chat-completions layer.

The layer is split by concern: :mod:`triagewright.llm.chat_wire` translates
``ModelRequest`` to the wire and the wire back to ``ModelResponse``, and
:mod:`triagewright.llm.chat_transport` holds :class:`OpenAICompatClient` (HTTP,
the retry policy, error mapping). Their names are re-exported here.

Gemini 3 rejects a tool-call turn sent back without its ``thought_signature``
(``tool_calls[].extra_content.google``). The contract has no field for it, so
:class:`GeminiClient` keeps the signatures of its own responses in a bounded
in-memory map keyed by tool-use id, name, and arguments, and resends them.
For history it didn't produce (a resumed recording, a restart), or a key two
responses signed differently, it sends Google's documented
``skip_thought_signature_validator`` on the first call of the turn.
Signatures are not part of ``ModelRequest``, so fingerprints don't change.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from triagewright.contracts.llm import ModelRequest, ModelResponse, Provider, ToolUseBlock
from triagewright.llm.chat_transport import _DURATION as _DURATION
from triagewright.llm.chat_transport import _ERROR_CODE as _ERROR_CODE
from triagewright.llm.chat_transport import MIN_ATTEMPT_S as MIN_ATTEMPT_S
from triagewright.llm.chat_transport import RETRYABLE_STATUS as RETRYABLE_STATUS
from triagewright.llm.chat_transport import OpenAICompatClient as OpenAICompatClient
from triagewright.llm.chat_transport import Sleep as Sleep
from triagewright.llm.chat_transport import _fits as _fits
from triagewright.llm.chat_transport import _json_or_none as _json_or_none
from triagewright.llm.chat_transport import _retry_info_seconds as _retry_info_seconds
from triagewright.llm.chat_transport import _status_error as _status_error
from triagewright.llm.chat_transport import log as log
from triagewright.llm.chat_transport import retry_after_seconds as retry_after_seconds
from triagewright.llm.chat_wire import _STOP_REASONS as _STOP_REASONS
from triagewright.llm.chat_wire import SKIP_THOUGHT_SIGNATURE as SKIP_THOUGHT_SIGNATURE
from triagewright.llm.chat_wire import SignatureKey as SignatureKey
from triagewright.llm.chat_wire import SignatureLookup as SignatureLookup
from triagewright.llm.chat_wire import _message_text as _message_text
from triagewright.llm.chat_wire import _usage as _usage
from triagewright.llm.chat_wire import _wire_messages as _wire_messages
from triagewright.llm.chat_wire import _wire_tool_choice as _wire_tool_choice
from triagewright.llm.chat_wire import from_chat_response as from_chat_response
from triagewright.llm.chat_wire import signature_key as signature_key
from triagewright.llm.chat_wire import thought_signatures as thought_signatures
from triagewright.llm.chat_wire import to_chat_request as to_chat_request
from triagewright.llm.chat_wire import tool_use_id as tool_use_id


class GeminiClient(OpenAICompatClient):
    """Gemini API free tier through its OpenAI-compatible endpoint."""

    provider = Provider.GEMINI
    max_signatures = 4096

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # ``None`` marks a key that two responses signed differently: send the sentinel.
        self._signatures: OrderedDict[SignatureKey, str | None] = OrderedDict()

    def _wire_body(self, request: ModelRequest) -> dict[str, Any]:
        return to_chat_request(request, signatures=lambda b: self._signatures.get(signature_key(b)))

    def _remember(self, data: Any, request: ModelRequest, response: ModelResponse) -> None:
        found = thought_signatures(data, request)
        for block in response.content:
            if not isinstance(block, ToolUseBlock) or block.id not in found:
                continue
            key, sig = signature_key(block), found[block.id]
            self._signatures[key] = sig if self._signatures.get(key, sig) == sig else None
            self._signatures.move_to_end(key)
        while len(self._signatures) > self.max_signatures:
            self._signatures.popitem(last=False)


class OpenRouterClient(OpenAICompatClient):
    """OpenRouter, ``:free`` models only."""

    provider = Provider.OPENROUTER
