"""
Tests for document ingestion, validation, and metadata extraction.
"""

import io
import pytest
from src.ingestion import (
    extract_pdf_pages,
    validate_file,
    clean_extracted_text,
    ingest_multiple_pdfs,
    IngestionResult
)

def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate a valid PDF in memory using pypdf or minimal writer."""
    try:
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject, ArrayObject, NumberObject
        writer = PdfWriter()
        for text in pages_text:
            writer.add_blank_page(width=612, height=792)
            # Add text or keep simple
        # Alternatively use reportlab or raw pdf if needed, or minimal valid PDF
        stream = io.BytesIO()
        writer.write(stream)
        stream.seek(0)
        return stream.getvalue()
    except Exception:
        # Minimal valid PDF string
        pdf_raw = (
            b"%PDF-1.4\n"
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
            b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>> endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer <</Size 4 /Root 1 0 R>>\nstartxref\n188\n%%EOF"
        )
        return pdf_raw


def test_text_cleaning_whitespace():
    raw = "This is a   sentence with   excessive    spaces.\n\n\n\nNext paragraph."
    cleaned = clean_extracted_text(raw)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Next paragraph." in cleaned


def test_text_cleaning_hyphenation():
    raw = "The meth-\nod was applied to analyze re-\nsearch data."
    cleaned = clean_extracted_text(raw)
    assert "method was applied" in cleaned
    assert "research data" in cleaned


def test_validate_file_extension():
    err = validate_file("test.txt", "test.txt")
    assert err is not None
    assert "only supports PDF" in err

    err_pdf = validate_file(b"content", "valid_paper.pdf")
    assert err_pdf is None


def test_validate_empty_file():
    empty_bytes = b""
    err = validate_file(empty_bytes, "empty_doc.pdf")
    assert err is not None
    assert "empty (0 bytes)" in err

    res = extract_pdf_pages(empty_bytes, "empty_doc.pdf")
    assert not res.is_success
    assert len(res.errors) > 0


def test_pdf_extraction_scanned_warning():
    # Minimal PDF without text content
    pdf_bytes = create_sample_pdf_bytes([""])
    res = extract_pdf_pages(pdf_bytes, "scanned_doc.pdf")
    assert res.is_success
    assert res.total_pages >= 1
    # Should detect scanned/empty warning
    assert any("scanned" in w.lower() or "selectable text" in w.lower() for w in res.warnings)


def test_multiple_pdf_ingestion():
    pdf1 = create_sample_pdf_bytes([""])
    pdf2 = create_sample_pdf_bytes([""])
    mapping = {"doc1.pdf": pdf1, "doc2.pdf": pdf2}
    results = ingest_multiple_pdfs(mapping)
    assert len(results) == 2
    assert results[0].doc_name == "doc1.pdf"
    assert results[1].doc_name == "doc2.pdf"
