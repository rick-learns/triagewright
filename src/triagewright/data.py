"""Loaders for the seed world, KB articles, and eval cases. Shared by every lane."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from triagewright.contracts.evals import EvalCase
from triagewright.contracts.rag import KBDocument
from triagewright.contracts.world import WorldState

_FRONT_MATTER = "---"


def load_world(seed_dir: Path) -> WorldState:
    """Load ``customers/accounts/orders/refunds/escalations.json`` from ``seed_dir``."""
    parts: dict[str, Any] = {}
    for name in WorldState.model_fields:
        path = seed_dir / f"{name}.json"
        if path.exists():
            parts[name] = json.loads(path.read_text(encoding="utf-8"))
    return WorldState.model_validate(parts)


def load_as_of(seed_dir: Path) -> datetime:
    """The seed world's fixed clock. Mock systems and policy checks use it instead of now()."""
    meta = json.loads((seed_dir / "meta.json").read_text(encoding="utf-8"))
    return datetime.fromisoformat(meta["as_of"])


def merge_world(base: WorldState, override: WorldState | None) -> WorldState:
    """Overlay ``override`` records on ``base`` by primary key (override wins)."""
    if override is None:
        return base
    keys = {
        "customers": "customer_id",
        "accounts": "customer_id",
        "orders": "order_id",
        "refunds": "refund_id",
        "escalations": "escalation_id",
    }
    merged: dict[str, list[Any]] = {}
    for field, key in keys.items():
        records = {getattr(r, key): r for r in getattr(base, field)}
        records.update({getattr(r, key): r for r in getattr(override, field)})
        merged[field] = list(records.values())
    return WorldState.model_validate(merged)


def parse_kb_markdown(text: str, *, fallback_id: str) -> KBDocument:
    """Parse a markdown article with YAML front matter into a :class:`KBDocument`."""
    if not text.startswith(_FRONT_MATTER):
        raise ValueError(f"{fallback_id}: missing YAML front matter")
    _, header, body = text.split(_FRONT_MATTER, 2)
    meta = yaml.safe_load(header) or {}
    if isinstance(meta.get("updated_at"), date):
        meta["updated_at"] = meta["updated_at"].isoformat()
    meta.setdefault("doc_id", fallback_id)
    return KBDocument.model_validate({**meta, "body": body.strip()})


def load_kb(kb_dir: Path) -> list[KBDocument]:
    return [
        parse_kb_markdown(p.read_text(encoding="utf-8"), fallback_id=p.stem)
        for p in sorted(kb_dir.glob("*.md"))
    ]


def load_eval_cases(cases_dir: Path) -> list[EvalCase]:
    """Load every ``*.yaml`` under ``cases_dir``. Raises on invalid or duplicate ids."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for path in sorted(cases_dir.rglob("*.yaml")):
        case = EvalCase.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if case.id in seen:
            raise ValueError(f"duplicate eval case id {case.id} in {path}")
        if path.stem != case.id:
            raise ValueError(f"{path}: file name must equal case id {case.id}")
        seen.add(case.id)
        cases.append(case)
    return cases


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
