"""
Tests for RAG pipeline, context formatting, citation verification, and evidence refusal.
"""

from src.vector_store import VectorStoreResult
from src.rag_pipeline import (
    RAGPipeline,
    format_context_for_prompt,
    extract_and_verify_citations,
    INSUFFICIENT_EVIDENCE_PHRASE
)

def test_format_context_for_prompt():
    sources = [
        VectorStoreResult(
            chunk_id="paper_p1_c0",
            doc_name="paper.pdf",
            page_number=1,
            text="First finding on solar cell durability.",
            similarity_score=0.88,
            distance=0.12,
            metadata={"file_name": "paper.pdf", "page_number": 1}
        )
    ]
    formatted = format_context_for_prompt(sources)
    assert "paper.pdf" in formatted
    assert "Page: 1" in formatted
    assert "First finding on solar cell durability." in formatted


def test_extract_and_verify_citations():
    sources = [
        VectorStoreResult(
            chunk_id="ai_research_p4_c1",
            doc_name="ai_research.pdf",
            page_number=4,
            text="Transformers demonstrate quadratic complexity.",
            similarity_score=0.85,
            distance=0.15,
            metadata={"file_name": "ai_research.pdf", "page_number": 4}
        )
    ]
    # Answer citing both real source and a fabricated hallucinated source
    answer = (
        "Transformers have quadratic complexity [ai_research.pdf, Page 4]. "
        "However, some claim linear complexity [hallucination.pdf, Page 99]."
    )
    citations = extract_and_verify_citations(answer, sources)
    assert len(citations) == 2

    c_valid = next(c for c in citations if c.doc_name == "ai_research.pdf")
    assert c_valid.page_number == 4
    assert c_valid.is_verified is True

    c_invalid = next(c for c in citations if c.doc_name == "hallucination.pdf")
    assert c_invalid.page_number == 99
    assert c_invalid.is_verified is False


def test_insufficient_evidence_refusal_empty_sources():
    pipeline = RAGPipeline()
    response = pipeline.answer_query("What is the battery efficiency?", sources=[])
    assert response.is_insufficient_evidence is True
    assert INSUFFICIENT_EVIDENCE_PHRASE in response.answer


def test_grounded_answer_generation():
    pipeline = RAGPipeline()
    sources = [
        VectorStoreResult(
            chunk_id="renewable_p2_c0",
            doc_name="renewable.pdf",
            page_number=2,
            text="Wind turbines installed in 2024 achieved an operational capacity factor of 42 percent.",
            similarity_score=0.82,
            distance=0.18,
            metadata={"file_name": "renewable.pdf", "page_number": 2}
        )
    ]
    response = pipeline.answer_query(
        "What was the wind turbine capacity factor in 2024?",
        sources=sources,
        force_fallback=True
    )
    assert response.is_insufficient_evidence is False
    assert len(response.citations) > 0
    assert response.citations[0].is_verified is True
    assert "42 percent" in response.answer
