"""
Tests for LangGraph agent workflow, evidence sufficiency grading, and query routing.
"""

from src.agent import ResearchAgent, WebSearchTool
from src.chunking import DocumentChunk, ChunkMetadata
from src.vector_store import ResearchVectorStore

def test_web_search_tool_separation():
    results = WebSearchTool.search("renewable energy grid integration")
    assert len(results) > 0
    item = results[0]
    assert "url" in item
    assert "retrieval_date" in item
    assert item["source_type"] == "External Web (Non-Document)"


def test_agent_workflow_sufficient_evidence():
    vs = ResearchVectorStore(persist_dir=":memory:", collection_name="agent_test", force_fallback_embedder=True)
    chunks = [
        DocumentChunk(
            text="Quantum computing processors using superconducting qubits achieved quantum supremacy on random circuit sampling.",
            metadata=ChunkMetadata(
                doc_id="quantum_paper.pdf",
                file_name="quantum_paper.pdf",
                page_number=3,
                chunk_id="quantum_p3_c0",
                chunk_index=0,
                char_start=0,
                char_end=110
            ),
            char_count=110,
            word_count=13
        )
    ]
    vs.add_chunks(chunks)

    agent = ResearchAgent(vector_store=vs)
    state = agent.run("What did superconducting qubits achieve in quantum computing?")

    assert state["is_sufficient"] is True
    assert "generation" in state
    assert len(state["retrieved_chunks"]) > 0
    assert any("Inspected query" in step for step in state["audit_trail"])


def test_agent_workflow_insufficient_evidence_gap_reporting():
    vs = ResearchVectorStore(persist_dir=":memory:", collection_name="agent_test_empty", force_fallback_embedder=True)
    # Empty vector store
    agent = ResearchAgent(vector_store=vs)
    state = agent.run("What is the nuclear magnetic resonance frequency of compound X?")

    assert state["is_sufficient"] is False
    assert "insufficient evidence" in state["generation"].lower()
    assert len(state["evidence_gaps"]) > 0
    assert state["retry_count"] >= 1  # Verify that query rewriting was attempted
