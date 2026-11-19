from __future__ import annotations

from pathlib import Path

import pytest

from triagewright.contracts.world import Customer, WorldState
from triagewright.data import load_eval_cases, merge_world, parse_kb_markdown
from triagewright.testing import REPO_ROOT, seed_as_of, seed_kb, seed_world


def test_seed_world_is_valid_and_consistent() -> None:
    world = seed_world()
    customer_ids = {c.customer_id for c in world.customers}
    order_ids = {o.order_id for o in world.orders}
    assert len(world.customers) >= 10 and len(world.orders) >= 20
    assert {a.customer_id for a in world.accounts} == customer_ids
    assert all(o.customer_id in customer_ids for o in world.orders)
    assert all(r.order_id in order_ids for r in world.refunds)
    assert all(c.email.endswith("@example.com") for c in world.customers)


def test_seed_clock_is_fixed() -> None:
    assert seed_as_of().isoformat() == "2026-06-15T12:00:00+00:00"


def test_kb_documents_parse_with_unique_ids() -> None:
    docs = seed_kb()
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)) >= 10
    assert all(Path(REPO_ROOT / "data" / "kb" / f"{d.doc_id}.md").exists() for d in docs)


def test_kb_front_matter_is_required() -> None:
    with pytest.raises(ValueError, match="front matter"):
        parse_kb_markdown("no header", fallback_id="kb_x")


def test_merge_world_overrides_by_primary_key() -> None:
    base = seed_world()
    changed = base.customers[0].model_copy(update={"tier": "vip"})
    added = Customer(
        customer_id="cus_new01",
        name="New Person",
        email="new.person@example.com",
        created_at=seed_as_of(),
    )
    merged = merge_world(base, WorldState(customers=[changed, added]))
    by_id = {c.customer_id: c for c in merged.customers}
    assert by_id[changed.customer_id].tier == "vip"
    assert "cus_new01" in by_id
    assert len(merged.customers) == len(base.customers) + 1
    assert merge_world(base, None) is base


def test_eval_cases_dir_loads() -> None:
    cases = load_eval_cases(REPO_ROOT / "evals" / "cases")
    assert len({c.id for c in cases}) == len(cases)


def test_eval_case_file_name_must_match_id(tmp_path: Path) -> None:
    (tmp_path / "wrong.yaml").write_text(
        "id: happy-001\ncategory: happy_path\ntitle: t\n"
        "ticket: {ticket_id: tkt_0001, body: hi, created_at: '2026-06-15T12:00:00Z'}\n"
        "expected: {status_any: [resolved], escalate: false}\n"
    )
    with pytest.raises(ValueError, match="file name"):
        load_eval_cases(tmp_path)
