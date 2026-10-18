"""
Document Ingestion & Validation Module.
Handles PDF text extraction, metadata extraction, validation of empty/encrypted files,
and graceful handling of scanned or image-only pages.
"""

import os
import io
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, BinaryIO

@dataclass
class DocumentPage:
    doc_name: str
    page_number: int  # 1-indexed
    text: str
    char_count: int
    word_count: int
    is_scanned_suspicion: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IngestionResult:
    doc_name: str
    total_pages: int
    valid_pages: int
    pages: List[DocumentPage]
    total_chars: int
    total_words: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    is_success: bool = True

    @property
    def has_text(self) -> bool:
        return self.total_chars > 0 and self.valid_pages > 0


def clean_extracted_text(text: str) -> str:
    """
    Cleans extracted raw text:
    - Normalizes null bytes and unicode artifacts
    - Recombines hyphenated line breaks (e.g. 'meth-\nod' -> 'method')
    - Collapses excessive whitespace while preserving paragraph structure
    """
    if not text:
        return ""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Fix hyphenated words at line endings
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Standardize newline representations
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple inline spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def validate_file(file_source: Union[str, Path, bytes, BinaryIO], doc_name: str) -> Optional[str]:
    """
    Validates file extension and basic size constraints before parsing.
    Returns error string if invalid, or None if valid.
    """
    if not doc_name.lower().endswith(".pdf"):
        return f"Unsupported file type for '{doc_name}'. ResearchPilot only supports PDF documents."
    
    if isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if not p.exists():
            return f"File '{doc_name}' does not exist at path: {file_source}"
        if p.stat().st_size == 0:
            return f"File '{doc_name}' is empty (0 bytes)."
    elif isinstance(file_source, bytes):
        if len(file_source) == 0:
            return f"Uploaded file '{doc_name}' is empty (0 bytes)."
    elif hasattr(file_source, "seek") and hasattr(file_source, "read"):
        file_source.seek(0, io.SEEK_END)
        size = file_source.tell()
        file_source.seek(0)
        if size == 0:
            return f"Uploaded stream for '{doc_name}' is empty (0 bytes)."
            
    return None


def extract_pdf_pages(
    file_source: Union[str, Path, bytes, BinaryIO],
    doc_name: str
) -> IngestionResult:
    """
    Extracts text and metadata page-by-page from a PDF document.
    Attempts PyMuPDF (fitz) first, with robust fallback to pypdf / pdfplumber.
    """
    val_err = validate_file(file_source, doc_name)
    if val_err:
        return IngestionResult(
            doc_name=doc_name,
            total_pages=0,
            valid_pages=0,
            pages=[],
            total_chars=0,
            total_words=0,
            errors=[val_err],
            is_success=False
        )

    # Normalize file_source to bytes or path
    file_bytes: Optional[bytes] = None
    file_path: Optional[str] = None
    if isinstance(file_source, (str, Path)):
        file_path = str(file_source)
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    elif hasattr(file_source, "read"):
        file_source.seek(0)
        file_bytes = file_source.read()

    # Try PyMuPDF (fitz)
    try:
        return _extract_with_fitz(file_path, file_bytes, doc_name)
    except ImportError:
        pass
    except Exception as e:
        # If fitz encountered an encryption or corruption error, log and try fallback
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            return IngestionResult(
                doc_name=doc_name,
                total_pages=0,
                valid_pages=0,
                pages=[],
                total_chars=0,
                total_words=0,
                errors=[f"PDF '{doc_name}' is encrypted or password-protected: {str(e)}"],
                is_success=False
            )

    # Fallback to pypdf
    try:
        return _extract_with_pypdf(file_path, file_bytes, doc_name)
    except ImportError:
        pass
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            return IngestionResult(
                doc_name=doc_name,
                total_pages=0,
                valid_pages=0,
                pages=[],
                total_chars=0,
                total_words=0,
                errors=[f"PDF '{doc_name}' is encrypted or password-protected: {str(e)}"],
                is_success=False
            )

    # Fallback to pdfplumber
    try:
        return _extract_with_pdfplumber(file_path, file_bytes, doc_name)
    except Exception as e:
        return IngestionResult(
            doc_name=doc_name,
            total_pages=0,
            valid_pages=0,
            pages=[],
            total_chars=0,
            total_words=0,
            errors=[f"Failed to extract text from '{doc_name}': {str(e)}"],
            is_success=False
        )


def _extract_with_fitz(
    file_path: Optional[str],
    file_bytes: Optional[bytes],
    doc_name: str
) -> IngestionResult:
    import fitz  # PyMuPDF

    if file_bytes is not None:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    else:
        doc = fitz.open(file_path)

    if doc.is_encrypted:
        doc.close()
        return IngestionResult(
            doc_name=doc_name,
            total_pages=0,
            valid_pages=0,
            pages=[],
            total_chars=0,
            total_words=0,
            errors=[f"PDF '{doc_name}' is encrypted or password-protected."],
            is_success=False
        )

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return IngestionResult(
            doc_name=doc_name,
            total_pages=0,
            valid_pages=0,
            pages=[],
            total_chars=0,
            total_words=0,
            errors=[f"PDF '{doc_name}' contains no pages."],
            is_success=False
        )

    pages: List[DocumentPage] = []
    warnings: List[str] = []
    total_chars = 0
    total_words = 0
    scanned_count = 0

    for idx, page in enumerate(doc):
        raw_text = page.get_text("text") or ""
        text = clean_extracted_text(raw_text)
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        is_scanned = char_count < 30
        if is_scanned:
            scanned_count += 1
            warnings.append(f"Page {idx + 1} has very low or no selectable text; might be scanned or image-only.")

        pages.append(DocumentPage(
            doc_name=doc_name,
            page_number=idx + 1,
            text=text,
            char_count=char_count,
            word_count=word_count,
            is_scanned_suspicion=is_scanned,
            metadata={
                "doc_name": doc_name,
                "page": idx + 1,
                "engine": "PyMuPDF",
                "width": page.rect.width,
                "height": page.rect.height
            }
        ))
        total_chars += char_count
        total_words += word_count

    doc.close()

    if total_chars == 0:
        warnings.append(f"All {total_pages} pages in '{doc_name}' are empty of extractable text. Scanned OCR required.")

    valid_pages = sum(1 for p in pages if p.char_count > 0)
    return IngestionResult(
        doc_name=doc_name,
        total_pages=total_pages,
        valid_pages=valid_pages,
        pages=pages,
        total_chars=total_chars,
        total_words=total_words,
        warnings=warnings,
        is_success=True
    )


def _extract_with_pypdf(
    file_path: Optional[str],
    file_bytes: Optional[bytes],
    doc_name: str
) -> IngestionResult:
    from pypdf import PdfReader

    stream = io.BytesIO(file_bytes) if file_bytes else open(file_path, "rb")
    reader = PdfReader(stream)

    if reader.is_encrypted:
        if file_bytes is None:
            stream.close()
        return IngestionResult(
            doc_name=doc_name,
            total_pages=0,
            valid_pages=0,
            pages=[],
            total_chars=0,
            total_words=0,
            errors=[f"PDF '{doc_name}' is encrypted or password-protected."],
            is_success=False
        )

    total_pages = len(reader.pages)
    if total_pages == 0:
        if file_bytes is None:
            stream.close()
        return IngestionResult(
            doc_name=doc_name,
            total_pages=0,
            valid_pages=0,
            pages=[],
            total_chars=0,
            total_words=0,
            errors=[f"PDF '{doc_name}' contains no pages."],
            is_success=False
        )

    pages: List[DocumentPage] = []
    warnings: List[str] = []
    total_chars = 0
    total_words = 0

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        text = clean_extracted_text(raw_text)
        char_count = len(text)
        word_count = len(text.split())
        is_scanned = char_count < 30
        if is_scanned:
            warnings.append(f"Page {idx + 1} has very low or no selectable text; might be scanned or image-only.")

        pages.append(DocumentPage(
            doc_name=doc_name,
            page_number=idx + 1,
            text=text,
            char_count=char_count,
            word_count=word_count,
            is_scanned_suspicion=is_scanned,
            metadata={"doc_name": doc_name, "page": idx + 1, "engine": "pypdf"}
        ))
        total_chars += char_count
        total_words += word_count

    if file_bytes is None:
        stream.close()

    valid_pages = sum(1 for p in pages if p.char_count > 0)
    return IngestionResult(
        doc_name=doc_name,
        total_pages=total_pages,
        valid_pages=valid_pages,
        pages=pages,
        total_chars=total_chars,
        total_words=total_words,
        warnings=warnings,
        is_success=True
    )


def _extract_with_pdfplumber(
    file_path: Optional[str],
    file_bytes: Optional[bytes],
    doc_name: str
) -> IngestionResult:
    import pdfplumber

    stream = io.BytesIO(file_bytes) if file_bytes else file_path
    with pdfplumber.open(stream) as pdf:
        total_pages = len(pdf.pages)
        pages: List[DocumentPage] = []
        warnings: List[str] = []
        total_chars = 0
        total_words = 0

        for idx, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            text = clean_extracted_text(raw_text)
            char_count = len(text)
            word_count = len(text.split())
            is_scanned = char_count < 30
            if is_scanned:
                warnings.append(f"Page {idx + 1} has very low text; might be scanned.")

            pages.append(DocumentPage(
                doc_name=doc_name,
                page_number=idx + 1,
                text=text,
                char_count=char_count,
                word_count=word_count,
                is_scanned_suspicion=is_scanned,
                metadata={"doc_name": doc_name, "page": idx + 1, "engine": "pdfplumber"}
            ))
            total_chars += char_count
            total_words += word_count

        valid_pages = sum(1 for p in pages if p.char_count > 0)
        return IngestionResult(
            doc_name=doc_name,
            total_pages=total_pages,
            valid_pages=valid_pages,
            pages=pages,
            total_chars=total_chars,
            total_words=total_words,
            warnings=warnings,
            is_success=True
        )


def ingest_multiple_pdfs(
    file_dict: Dict[str, Union[str, Path, bytes, BinaryIO]]
) -> List[IngestionResult]:
    """
    Convenience method to ingest multiple PDF files from a mapping of {doc_name: file_source}.
    """
    results = []
    for doc_name, file_source in file_dict.items():
        res = extract_pdf_pages(file_source, doc_name)
        results.append(res)
    return results
