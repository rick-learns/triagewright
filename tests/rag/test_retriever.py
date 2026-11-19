from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from triagewright.config import Settings
from triagewright.contracts.common import canonical_json
from triagewright.contracts.ports import Retriever
from triagewright.contracts.rag import Chunk, MetadataFilter, RetrievalQuery
from triagewright.rag import HashingEmbedder, HybridRetriever, KBIndex, build_retriever
from triagewright.rag.index import rank_scores
from triagewright.rag.retriever import Mode
from triagewright.testing import REPO_ROOT, seed_kb

MODES: list[Mode] = ["keyword", "vector", "hybrid"]

# Questions with one clearly right article, answered by every method.
KNOWN = [
    ("How many days do I have to get a refund?", "kb_refund_policy"),
    ("my package is late, where is tracking", "kb_shipping"),
    ("I forgot my password and my account is locked", "kb_account_lockout"),
    ("I was charged twice", "kb_billing"),
    ("delete my personal data", "kb_privacy"),
    ("item arrived broken", "kb_damaged_items"),
    ("can I cancel my order", "kb_order_changes"),
    ("talk to a human", "kb_contact_escalation"),
]


def search(index: KBIndex, text: str, mode: Mode = "hybrid", **kw: object) -> list[str]:
    query = RetrievalQuery.model_validate({"text": text, **kw})
    return [r.chunk.chunk_id for r in HybridRetriever(index, mode=mode).search(query)]


def test_is_a_retriever(kb_index: KBIndex) -> None:
    assert isinstance(HybridRetriever(kb_index), Retriever)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("text", "doc_id"), KNOWN)
def test_known_questions_find_their_article(
    kb_index: KBIndex, mode: Mode, text: str, doc_id: str
) -> None:
    assert search(kb_index, text, mode)[0].startswith(f"{doc_id}#")


def test_refund_window_question_hits_the_window_chunk(kb_index: KBIndex) -> None:
    for mode in MODES:
        hits = search(kb_index, "How many days do I have to get a refund?", mode, k=2)
        assert "kb_refund_policy#0" in hits  # "within 30 days of the delivery date"


def test_hybrid_fuses_both_rankings(kb_index: KBIndex) -> None:
    text = "how long do refunds take to reach my card"
    keyword = search(kb_index, text, "keyword")
    vector = search(kb_index, text, "vector")
    hybrid = search(kb_index, text, "hybrid", k=20)
    assert keyword[0] != vector[0]
    union = set(search(kb_index, text, "keyword", k=20)) | set(
        search(kb_index, text, "vector", k=20)
    )
    assert set(hybrid) <= union and len(hybrid) == min(20, len(union))
    assert hybrid[0] in {keyword[0], vector[0]}


@pytest.mark.parametrize("mode", MODES)
def test_results_are_ranked_and_labeled(kb_index: KBIndex, mode: Mode) -> None:
    results = HybridRetriever(kb_index, mode=mode).search(RetrievalQuery(text="refund", k=4))
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert all(r.method == mode and r.score > 0 for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


@pytest.mark.parametrize("mode", MODES)
def test_category_filter(kb_index: KBIndex, mode: Mode) -> None:
    flt = MetadataFilter(categories=["refunds"])
    hits = search(kb_index, "my package was delivered late", mode, k=20, filters=flt)
    assert hits and all(h.split("#")[0] in {"kb_refund_policy", "kb_damaged_items"} for h in hits)


@pytest.mark.parametrize("mode", MODES)
def test_tag_and_date_filters(kb_index: KBIndex, mode: Mode) -> None:
    by_tag = search(kb_index, "refund", mode, k=20, filters=MetadataFilter(tags_any=["rma"]))
    assert by_tag and {h.split("#")[0] for h in by_tag} == {"kb_returns"}
    # Strictly after: kb_refund_policy (2026-05-01) is excluded at its own date.
    recent = MetadataFilter(updated_after=date(2026, 5, 1))
    hits = {h.split("#")[0] for h in search(kb_index, "refund", mode, k=20, filters=recent)}
    assert "kb_damaged_items" in hits and "kb_refund_policy" not in hits
    none = MetadataFilter(categories=["no-such-category"])
    assert search(kb_index, "refund", mode, filters=none) == []


@pytest.mark.parametrize("mode", MODES)
def test_k_and_min_score(kb_index: KBIndex, mode: Mode) -> None:
    assert len(search(kb_index, "refund order delivery", mode, k=2)) == 2
    assert search(kb_index, "refund", mode, min_score=1e9) == []


@pytest.mark.parametrize("mode", MODES)
def test_stopword_only_query_returns_nothing(kb_index: KBIndex, mode: Mode) -> None:
    assert search(kb_index, "what is the", mode) == []


@pytest.mark.parametrize("mode", MODES)
def test_untrusted_chunks_are_returned_for_screening(kb_index: KBIndex, mode: Mode) -> None:
    query = RetrievalQuery(text="how do I pair the Pebble Speaker", k=1)
    (top,) = HybridRetriever(kb_index, mode=mode).search(query)
    assert top.chunk.doc_id == "kb_community_tips" and top.chunk.trusted is False


def test_ties_break_on_doc_then_ordinal() -> None:
    def chunk(doc_id: str, ordinal: int) -> Chunk:
        return Chunk(
            chunk_id=f"{doc_id}#{ordinal}",
            doc_id=doc_id,
            ordinal=ordinal,
            title="t",
            category="c",
            text="x",
            start_char=0,
            end_char=1,
        )

    scored = [(chunk("kb_bb", 0), 0.5), (chunk("kb_aa", 10), 0.5)]
    scored += [
        (chunk("kb_aa", 2), 0.5000000001),
        (chunk("kb_cc", 0), 0.0),
        (chunk("kb_cc", 1), 0.9),
    ]
    ranked = [c.chunk_id for c, _ in rank_scores(scored)]
    assert ranked == ["kb_cc#1", "kb_aa#2", "kb_aa#10", "kb_bb#0"]


def test_results_are_identical_across_indexes(tmp_path: Path, kb_index: KBIndex) -> None:
    queries = [RetrievalQuery(text=t, k=10) for t, _ in KNOWN]
    with KBIndex(tmp_path / "kb.sqlite", HashingEmbedder()) as other:
        other.ingest(list(reversed(seed_kb())))
        for mode in MODES:
            a = [HybridRetriever(kb_index, mode=mode).search(q) for q in queries]
            b = [HybridRetriever(other, mode=mode).search(q) for q in queries]
            dump = [[r.model_dump(mode="json") for r in rs] for rs in a]
            assert canonical_json(dump) == canonical_json(
                [[r.model_dump(mode="json") for r in rs] for rs in b]
            )


@pytest.mark.anyio
async def test_build_retriever_from_settings(tmp_path: Path) -> None:
    settings = Settings(db_path=tmp_path / "var" / "tw.sqlite", kb_dir=REPO_ROOT / "data" / "kb")
    first = build_retriever(settings)
    first.index.close()
    second = build_retriever(settings)
    try:
        assert second.index.ingest(seed_kb()).unchanged == len(seed_kb())
        results = await second.retrieve(RetrievalQuery(text="Can I get my money back?"))
        assert results[0].chunk.doc_id == "kb_refund_policy"
        assert results[0].method == "hybrid"
    finally:
        second.index.close()
