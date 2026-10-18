"""
Tests for text chunking, recursive boundary splitting, and metadata preservation.
"""

from src.ingestion import DocumentPage
from src.chunking import (
    chunk_document_page,
    chunk_document_pages,
    get_chunk_statistics,
    _split_text_recursive
)

def test_recursive_split_short_text():
    text = "Short sentence."
    splits = _split_text_recursive(text, chunk_size=100, chunk_overlap=20)
    assert len(splits) == 1
    assert splits[0][0] == "Short sentence."


def test_recursive_split_long_text_overlap():
    text = (
        "Paragraph one discusses renewable solar technologies and efficiency metrics. "
        "It provides benchmarks across various testing facilities.\n\n"
        "Paragraph two examines wind energy grid integration challenges. "
        "Storage mechanisms such as lithium and flow batteries are compared."
    )
    splits = _split_text_recursive(text, chunk_size=80, chunk_overlap=20)
    assert len(splits) > 1
    # Check that chunks have content
    for txt, start, end in splits:
        assert len(txt) > 0
        assert end > start


def test_chunk_document_page_metadata():
    page = DocumentPage(
        doc_name="solar_study.pdf",
        page_number=3,
        text="Solar energy efficiency increased to 24.5 percent under optimal concentrated photovoltaic cells. " * 5,
        char_count=500,
        word_count=70
    )
    chunks = chunk_document_page(page, chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.metadata.file_name == "solar_study.pdf"
        assert c.metadata.page_number == 3
        assert c.metadata.chunk_index == i
        assert f"p3_c{i}" in c.metadata.chunk_id
        assert c.char_count == len(c.text)


def test_chunk_statistics_helper():
    page = DocumentPage(
        doc_name="paper.pdf",
        page_number=1,
        text="A detailed description of the experiments and materials utilized in the study.",
        char_count=77,
        word_count=11
    )
    chunks = chunk_document_pages([page], chunk_size=50, chunk_overlap=10)
    stats = get_chunk_statistics(chunks)
    assert stats["total_chunks"] == len(chunks)
    assert stats["avg_chars"] > 0
    assert "paper.pdf" in stats["documents_represented"]
