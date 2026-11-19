# ADR-006: Span errors never carry exception messages

- Status: Accepted (PR #20)
- Date: 2026-11-19

## Context

Traces are written to `var/traces/*.jsonl` or sent over OTLP, and they outlive
the request. An exception's message can quote what caused it: ticket text,
tool arguments, or a provider's error text echoing the prompt. The telemetry
span wrapper used to record an unwinding exception's message as the span
status. A failed model call, on the other hand, showed only `status: error`, so
live failures couldn't be told apart in traces.

## Decision

- When a span unwinds with an exception and the caller hasn't set an error,
  `OTelRequestTelemetry.span` (`triagewright.observability.telemetry`) sets the
  status description to the exception's type name only.
- An error the caller sets explicitly (`SpanHandle.set_error`) is not
  overwritten by the unwinding exception.
- A failed model call sets its span error in `RequestRun.call_model`
  (`triagewright.agent.runtime`) to the exception type plus the HTTP status
  when the error carries one, for example `ModelUnavailableError (HTTP 429)`.
  Timeouts and refused pacing set `timeout` and `pacing`.
- Tool spans set the error to the tool's `ErrorCategory` value.
- The JSONL exporter keeps the span's status description as the `error`
  attribute (`triagewright.observability.exporters`).
- The model adapters keep the same rule for their own messages: provider,
  model, HTTP status, and the provider's status code, never the provider's
  free text (`triagewright.llm.openai_compat`). The API logs only an unhandled
  exception's type.

## Consequences

- Traces say what kind of failure happened (quota, overload, auth, timeout)
  without copying customer content.
- Debugging a specific failure needs a reproduction or the replay recording,
  not the trace alone.
- New error paths must follow the same rule: set a type, status, or category,
  never `str(exc)`.

## Alternatives

- **Record the full message.** Rejected: messages can hold ticket text and PII.
- **Redact messages with `Guardrails.redact`.** Rejected: redaction targets
  known PII patterns, not arbitrary text such as an injected instruction or an
  order history.
- **Record nothing beyond `error`.** The previous behavior. Rejected: live
  failures couldn't be diagnosed.
