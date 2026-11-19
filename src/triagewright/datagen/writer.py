"""Writing a :class:`Generated` world and its tickets to a directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triagewright.contracts.common import canonical_json
from triagewright.contracts.world import WorldState
from triagewright.datagen.models import Generated


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write(gen: Generated, out: Path) -> list[Path]:
    """Write the world files, ``meta.json``, ``tickets.jsonl``, and ``expected.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {
        f"{name}.json": _json([r.model_dump(mode="json") for r in getattr(gen.world, name)])
        for name in WorldState.model_fields
    }
    files["meta.json"] = _json(gen.meta)
    files["tickets.jsonl"] = "".join(
        canonical_json(t.model_dump(mode="json")) + "\n" for t in gen.tickets
    )
    files["expected.jsonl"] = "".join(
        canonical_json(e.model_dump(mode="json")) + "\n" for e in gen.expected
    )
    paths = []
    for name, text in sorted(files.items()):
        path = out / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
