from __future__ import annotations

from datetime import date

from triagewright.contracts.rag import KBDocument
from triagewright.rag import chunk_document
from triagewright.testing import seed_kb


def doc(body: str, **kw: object) -> KBDocument:
    fields: dict[str, object] = {
        "doc_id": "kb_test",
        "title": "Test",
        "category": "testing",
        "tags": ["a", "b"],
        "updated_at": date(2026, 1, 1),
        "body": body,
    }
    return KBDocument.model_validate(fields | kw)


def test_every_kb_chunk_is_a_verbatim_slice_with_metadata() -> None:
    for d in seed_kb():
        chunks = chunk_document(d)
        assert chunks
        for ordinal, c in enumerate(chunks):
            assert c.chunk_id == f"{d.doc_id}#{ordinal}" and c.ordinal == ordinal
            assert d.body[c.start_char : c.end_char] == c.text
            assert (c.doc_id, c.title, c.category, c.tags, c.trusted) == (
                d.doc_id,
                d.title,
                d.category,
                d.tags,
                d.trusted,
            )


def test_one_chunk_per_heading_section() -> None:
    refund = next(d for d in seed_kb() if d.doc_id == "kb_refund_policy")
    chunks = chunk_document(refund)
    assert [c.text.splitlines()[0] for c in chunks] == [
        "## Who can get a refund",
        "## Limits",
        "## Approval",
        "## Not refundable",
    ]
    assert "within 30 days of the delivery date" in chunks[0].text


def test_untrusted_flag_is_kept() -> None:
    tips = next(d for d in seed_kb() if d.doc_id == "kb_community_tips")
    assert all(not c.trusted for c in chunk_document(tips))


def test_long_sections_split_between_paragraphs() -> None:
    body = "Intro line.\n\n## Part\n\n" + "\n\n".join(f"Para {i} " + "x" * 30 for i in range(4))
    chunks = chunk_document(doc(body), max_chars=90)
    assert chunks[0].text == "Intro line."
    assert chunks[1].text.startswith("## Part\n\nPara 0")
    assert all(len(c.text) <= 90 for c in chunks)
    assert "".join(c.text for c in chunks).count("Para") == 4


def test_oversized_paragraph_stays_whole() -> None:
    chunks = chunk_document(doc("y" * 50), max_chars=10)
    assert [(c.start_char, c.end_char) for c in chunks] == [(0, 50)]


def test_chunking_is_deterministic() -> None:
    assert [chunk_document(d) for d in seed_kb()] == [chunk_document(d) for d in seed_kb()]
