"""Typed model-client errors.

Messages carry the provider, model, HTTP status, and the provider's short error
text, never API keys or prompt text. ``retryable`` says whether the condition is
transient (quota, overload, network); the client has already spent its own
retry budget when it raises, so a caller that retries does so knowingly.
"""

from __future__ import annotations


class ModelError(Exception):
    retryable: bool = False

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ModelConfigError(ModelError):
    """Bad setup: a non-free model, a missing key, or a request for another model."""


class ModelAuthError(ModelError):
    """401/403: the key is missing, wrong, or lacks access."""


class ModelRequestError(ModelError):
    """Any other 4xx: the provider rejected the request as sent."""


class ModelUnavailableError(ModelError):
    """429, 5xx, timeout, or connection failure that outlasted the retry budget."""

    retryable = True

    def __init__(
        self, message: str, *, status: int | None = None, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message, status=status)
        self.retry_after_s = retry_after_s


class ModelResponseError(ModelError):
    """The provider answered, but not with something we can translate."""


class ReplayMissError(ModelError):
    """No unserved recording matches the request. Replay never falls back to a live call."""
