"""Export JSON Schemas for the public contracts into ``schemas/``.

Run after any contract change: ``uv run python scripts/export_schemas.py``.
``--check`` exits 1 if the committed schemas are stale (used in CI via tests).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from triagewright.contracts import agents, evals, guardrails, llm, rag, telemetry, tickets, tools
from triagewright.contracts.common import CONTRACT_VERSION
from triagewright.contracts.world import WorldState

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"

EXPORTED: dict[str, type[BaseModel]] = {
    "ticket": tickets.Ticket,
    "world_state": WorldState,
    "tool_result": tools.ToolResult,
    "tool_error": tools.ToolError,
    "tool_call_record": tools.ToolCallRecord,
    "refund_proposal": tools.RefundProposal,
    "refund_eligibility": tools.RefundEligibility,
    "triage_result": agents.TriageResult,
    "answer_draft": agents.AnswerDraft,
    "action_result": agents.ActionResult,
    "escalation_decision": agents.EscalationDecision,
    "final_response": agents.FinalResponse,
    "citation": rag.Citation,
    "kb_document": rag.KBDocument,
    "guardrail_verdict": guardrails.GuardrailVerdict,
    "approval_request": guardrails.ApprovalRequest,
    "approval_decision": guardrails.ApprovalDecision,
    "model_request": llm.ModelRequest,
    "model_response": llm.ModelResponse,
    "recording": llm.Recording,
    "trace_event": telemetry.TraceEvent,
    "request_metrics": telemetry.RequestMetrics,
    "eval_case": evals.EvalCase,
    "eval_run_summary": evals.EvalRunSummary,
}

TOOL_INPUTS = {name.value: spec.input_model for name, spec in tools.TOOL_SPECS.items()}


def render() -> dict[str, str]:
    files: dict[str, str] = {}
    for name, model in EXPORTED.items():
        schema = model.model_json_schema()
        schema["$id"] = f"triagewright/{name}/{CONTRACT_VERSION}"
        files[f"{name}.schema.json"] = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    for name, model in TOOL_INPUTS.items():
        schema = model.model_json_schema()
        schema["$id"] = f"triagewright/tools/{name}/{CONTRACT_VERSION}"
        files[f"tools/{name}.input.schema.json"] = (
            json.dumps(schema, indent=2, sort_keys=True) + "\n"
        )
    return files


def main(argv: list[str]) -> int:
    files = render()
    if "--check" in argv:
        stale = [
            n
            for n, body in files.items()
            if not (SCHEMA_DIR / n).is_file()
            or (SCHEMA_DIR / n).read_text(encoding="utf-8") != body
        ]
        extra = {str(p.relative_to(SCHEMA_DIR)) for p in SCHEMA_DIR.rglob("*.json")} - set(files)
        for n in [*stale, *sorted(extra)]:
            print(f"stale schema: {n}", file=sys.stderr)
        return 1 if stale or extra else 0
    for n, body in files.items():
        path = SCHEMA_DIR / n
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    print(f"wrote {len(files)} schemas to {SCHEMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
