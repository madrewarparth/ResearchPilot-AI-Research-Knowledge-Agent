"""
End-to-End Integration Test for ResearchPilot.
Tests complete pipeline: PDF Ingestion -> Text Chunking -> Chroma Indexing -> Similarity Retrieval -> Grounded Answer Generation.
"""

from pathlib import Path
from src.ingestion import extract_pdf_pages
from src.chunking import chunk_document_pages
from src.vector_store import ResearchVectorStore
from src.rag_pipeline import RAGPipeline
from src.agent import ResearchAgent

def test_full_pipeline_end_to_end():
    sample_pdf = Path("data/sample_renewable_study.pdf")
    assert sample_pdf.exists(), "Sample PDF fixture should exist"

    # 1. Ingestion
    ingest_res = extract_pdf_pages(sample_pdf, "sample_renewable_study.pdf")
    assert ingest_res.is_success
    assert ingest_res.total_pages == 2
    assert ingest_res.total_chars > 200

    # 2. Chunking
    chunks = chunk_document_pages(ingest_res.pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(c.metadata.file_name == "sample_renewable_study.pdf" for c in chunks)

    # 3. Vector Storage
    vs = ResearchVectorStore(persist_dir=":memory:", collection_name="e2e_test", force_fallback_embedder=True)
    indexed_count = vs.add_chunks(chunks)
    assert indexed_count == len(chunks)

    # 4. Retrieval & Agent Run
    agent = ResearchAgent(vector_store=vs)
    state = agent.run("What was the laboratory efficiency of perovskite solar cells?")

    assert state["is_sufficient"] is True
    assert "25.7%" in state["generation"]
    assert len(state["citations"]) > 0
    assert state["citations"][0]["is_verified"] is True
    assert state["citations"][0]["doc_name"] == "sample_renewable_study.pdf"
    assert state["citations"][0]["page_number"] == 1
