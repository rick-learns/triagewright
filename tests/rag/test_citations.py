from __future__ import annotations

from triagewright.contracts.rag import Chunk, Citation
from triagewright.contracts.tools import KBHit
from triagewright.rag import KBIndex, make_citation, verify_citations


def chunk(index: KBIndex, chunk_id: str) -> Chunk:
    found = index.get_chunk(chunk_id)
    assert found is not None
    return found


def test_verbatim_quote_from_trusted_chunk_is_citable(kb_index: KBIndex) -> None:
    source = chunk(kb_index, "kb_refund_policy#0")
    citation = make_citation(source, "  within 30 days of the delivery date ")
    assert citation == Citation(
        doc_id="kb_refund_policy",
        chunk_id="kb_refund_policy#0",
        title="Refund policy",
        quote="within 30 days of the delivery date",
    )


def test_paraphrase_empty_and_overlong_quotes_are_not_citable(kb_index: KBIndex) -> None:
    source = chunk(kb_index, "kb_refund_policy#0")
    assert make_citation(source, "within thirty days of delivery") is None
    assert make_citation(source, "   ") is None
    assert make_citation(source, source.text + "x" * 400) is None


def test_untrusted_chunk_is_never_citable(kb_index: KBIndex) -> None:
    source = chunk(kb_index, "kb_community_tips#0")
    assert make_citation(source, "Pair the Pebble Speaker") is None


def test_verify_citations_filters_candidates(kb_index: KBIndex) -> None:
    refund = chunk(kb_index, "kb_refund_policy#0")
    tips = chunk(kb_index, "kb_community_tips#0")
    good = Citation(
        doc_id=refund.doc_id, chunk_id=refund.chunk_id, title="spoofed", quote="Delivered orders"
    )
    candidates = [
        good,
        good,  # duplicate
        good.model_copy(update={"quote": "made up policy"}),
        good.model_copy(update={"doc_id": "kb_shipping"}),  # wrong doc for the chunk
        good.model_copy(update={"chunk_id": "kb_refund_policy#9"}),  # not retrieved
        Citation(doc_id=tips.doc_id, chunk_id=tips.chunk_id, title=tips.title, quote="Pair"),
    ]
    kept = verify_citations(candidates, [refund, tips])
    assert kept == [good.model_copy(update={"title": "Refund policy"})]


def test_works_with_search_kb_hits(kb_index: KBIndex) -> None:
    source = chunk(kb_index, "kb_shipping#0")
    hit = KBHit(
        **source.model_dump(include={"doc_id", "chunk_id", "title", "category", "trusted", "text"}),
        score=1.0,
    )
    quote = "Standard shipping takes 3 to 7 business days"
    candidate = Citation(doc_id=hit.doc_id, chunk_id=hit.chunk_id, title=hit.title, quote=quote)
    assert verify_citations([candidate], [hit]) == [candidate]
