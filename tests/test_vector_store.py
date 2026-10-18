"""
Tests for vector store indexing, similarity search, and citation metadata retrieval.
"""

import tempfile
import shutil
from src.chunking import DocumentChunk, ChunkMetadata
from src.vector_store import ResearchVectorStore, FallbackDeterministicEmbedder

def test_fallback_embedder_dimension_and_normalization():
    embedder = FallbackDeterministicEmbedder()
    vec = embedder.embed_text("Renewable solar energy and power grids.")
    assert len(vec) == 256
    norm = sum(v * v for v in vec)
    # L2 normalized vectors have norm close to 1.0
    assert abs(norm - 1.0) < 1e-4


def test_vector_store_indexing_and_search():
    temp_dir = tempfile.mkdtemp()
    try:
        vs = ResearchVectorStore(persist_dir=temp_dir, collection_name="test_col", force_fallback_embedder=True)
        chunks = [
            DocumentChunk(
                text="Perovskite solar cell efficiency reached 25.7% in laboratory testing environments.",
                metadata=ChunkMetadata(
                    doc_id="solar_tech.pdf",
                    file_name="solar_tech.pdf",
                    page_number=2,
                    chunk_id="solar_tech_p2_c0",
                    chunk_index=0,
                    char_start=0,
                    char_end=85
                ),
                char_count=85,
                word_count=10
            ),
            DocumentChunk(
                text="Wind turbine aerodynamic blade pitch optimization reduces mechanical stress in gusts.",
                metadata=ChunkMetadata(
                    doc_id="wind_power.pdf",
                    file_name="wind_power.pdf",
                    page_number=5,
                    chunk_id="wind_power_p5_c0",
                    chunk_index=0,
                    char_start=0,
                    char_end=86
                ),
                char_count=86,
                word_count=11
            )
        ]
        indexed = vs.add_chunks(chunks)
        assert indexed == 2

        results = vs.similarity_search("perovskite solar efficiency", top_k=2)
        assert len(results) > 0
        top = results[0]
        assert top.doc_name == "solar_tech.pdf"
        assert top.page_number == 2
        assert "[solar_tech.pdf, Page 2" in top.format_citation()

        # Test document filtering
        filtered = vs.similarity_search("blade turbine", top_k=2, filter_doc="wind_power.pdf")
        assert len(filtered) > 0
        assert all(r.doc_name == "wind_power.pdf" for r in filtered)

        # Test stats
        stats = vs.get_collection_stats()
        assert stats["total_chunks"] == 2
        assert "solar_tech.pdf" in stats["distinct_documents"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
