"""
Text Chunking & Splitting Engine.
Splits extracted document text into overlapping chunks while preserving rich provenance metadata
(document name, page number, character offsets, chunk index).
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from src.ingestion import DocumentPage, IngestionResult

@dataclass
class ChunkMetadata:
    doc_id: str
    file_name: str
    page_number: int  # 1-indexed
    chunk_id: str
    chunk_index: int
    char_start: int
    char_end: int
    total_doc_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DocumentChunk:
    text: str
    metadata: ChunkMetadata
    char_count: int
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "metadata": self.metadata.to_dict()
        }


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None
) -> List[Tuple[str, int, int]]:
    """
    Recursively splits text into segments conforming to chunk_size and chunk_overlap,
    respecting structural boundaries (paragraphs, sentences, spaces).
    Returns list of (chunk_text, start_offset, end_offset).
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    if not text or len(text.strip()) == 0:
        return []

    # If text fits in one chunk, return it immediately
    if len(text) <= chunk_size:
        return [(text.strip(), 0, len(text))]

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If not at the end of the text, look for a separator to split gracefully
        if end < text_len:
            best_split = -1
            # Search backward from `end` down to `start + chunk_overlap`
            for sep in separators:
                if not sep:
                    continue
                pos = text.rfind(sep, start + chunk_overlap, end)
                if pos != -1:
                    best_split = pos + len(sep)
                    break
            
            if best_split != -1:
                end = best_split

        segment = text[start:end]
        stripped = segment.strip()
        if stripped:
            # Calculate real offsets for stripped text
            leading_ws = len(segment) - len(segment.lstrip())
            trailing_ws = len(segment) - len(segment.rstrip())
            seg_start = start + leading_ws
            seg_end = end - trailing_ws
            chunks.append((stripped, seg_start, seg_end))

        if end >= text_len:
            break

        # Move forward, stepping back by chunk_overlap
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_document_page(
    page: DocumentPage,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 30
) -> List[DocumentChunk]:
    """
    Splits a single DocumentPage into DocumentChunks, embedding exact page provenance.
    """
    if not page.text or len(page.text.strip()) < min_chunk_chars:
        return []

    splits = _split_text_recursive(page.text, chunk_size, chunk_overlap)
    chunks: List[DocumentChunk] = []

    # Sanitize doc name for chunk ID
    clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', page.doc_name)

    for idx, (txt, c_start, c_end) in enumerate(splits):
        if len(txt) < min_chunk_chars:
            continue
        chunk_id = f"{clean_name}_p{page.page_number}_c{idx}"
        meta = ChunkMetadata(
            doc_id=page.doc_name,
            file_name=page.doc_name,
            page_number=page.page_number,
            chunk_id=chunk_id,
            chunk_index=idx,
            char_start=c_start,
            char_end=c_end,
            total_doc_chunks=len(splits)
        )
        chunks.append(DocumentChunk(
            text=txt,
            metadata=meta,
            char_count=len(txt),
            word_count=len(txt.split())
        ))

    return chunks


def chunk_document_pages(
    pages: List[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 30
) -> List[DocumentChunk]:
    """
    Splits a list of DocumentPages into DocumentChunks and updates total chunk counts per doc.
    """
    all_chunks: List[DocumentChunk] = []
    doc_chunk_counts: Dict[str, int] = {}

    for page in pages:
        p_chunks = chunk_document_page(page, chunk_size, chunk_overlap, min_chunk_chars)
        all_chunks.extend(p_chunks)
        doc_chunk_counts[page.doc_name] = doc_chunk_counts.get(page.doc_name, 0) + len(p_chunks)

    # Update total_doc_chunks
    for chunk in all_chunks:
        chunk.metadata.total_doc_chunks = doc_chunk_counts.get(chunk.metadata.file_name, len(all_chunks))

    return all_chunks


def chunk_ingestion_results(
    results: List[IngestionResult],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 30
) -> List[DocumentChunk]:
    """
    Processes all successfully ingested pages from a list of IngestionResults into chunks.
    """
    pages_to_chunk: List[DocumentPage] = []
    for res in results:
        if res.is_success and res.pages:
            pages_to_chunk.extend(res.pages)
    return chunk_document_pages(pages_to_chunk, chunk_size, chunk_overlap, min_chunk_chars)


def get_chunk_statistics(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """
    Returns diagnostic statistics on the generated chunks.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "documents_represented": []
        }

    char_lens = [c.char_count for c in chunks]
    docs = sorted(list({c.metadata.file_name for c in chunks}))

    return {
        "total_chunks": len(chunks),
        "avg_chars": round(sum(char_lens) / len(char_lens), 1),
        "min_chars": min(char_lens),
        "max_chars": max(char_lens),
        "documents_represented": docs
    }
