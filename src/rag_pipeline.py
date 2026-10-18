"""
RAG Pipeline & Answer Generation.
Constructs strictly grounded prompt templates, calls local Ollama LLM (llama3.2:3b),
parses and verifies citations against retrieved sources, and handles insufficient evidence.
"""

import re
import time
import json
import urllib.request
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from config import settings
from src.vector_store import VectorStoreResult

INSUFFICIENT_EVIDENCE_PHRASE = (
    "Based on the uploaded documents, there is insufficient evidence to answer this question."
)

SYSTEM_PROMPT = """You are ResearchPilot, a meticulous AI research assistant.
Your answers must be STRICTLY GROUNDED in the provided document passages.

CRITICAL RULES:
1. ONLY state facts that are explicitly mentioned in the provided Context passages.
2. For EVERY claim or finding you make, cite the source document and page number in brackets: [filename.pdf, Page X].
3. If the provided passages DO NOT contain sufficient evidence to answer the question, you MUST reply:
   "Based on the uploaded documents, there is insufficient evidence to answer this question."
   Do NOT guess, fabricate, or speculate beyond the supplied evidence.
4. When comparing multiple documents, highlight points of agreement, divergence, and differences in methodology.
"""

USER_PROMPT_TEMPLATE = """Context Passages:
{context}

Question:
{question}

Provide a comprehensive, factual answer citing sources in the format [filename.pdf, Page X]:"""


@dataclass
class Citation:
    doc_name: str
    page_number: int
    raw_citation: str
    is_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "raw_citation": self.raw_citation,
            "is_verified": self.is_verified
        }


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[VectorStoreResult]
    citations: List[Citation]
    is_insufficient_evidence: bool
    model_used: str
    latency_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.metadata for s in self.sources],
            "citations": [c.to_dict() for c in self.citations],
            "is_insufficient_evidence": self.is_insufficient_evidence,
            "model_used": self.model_used,
            "latency_seconds": self.latency_seconds
        }


def format_context_for_prompt(sources: List[VectorStoreResult]) -> str:
    """
    Formats retrieved vector store results into clear, delineated passages for LLM prompting.
    """
    if not sources:
        return "No relevant passages found."

    context_blocks = []
    for idx, s in enumerate(sources, 1):
        block = (
            f"--- PASSAGE {idx} ---\n"
            f"Document: {s.doc_name}\n"
            f"Page: {s.page_number}\n"
            f"Chunk ID: {s.chunk_id}\n"
            f"Content:\n{s.text}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)


def extract_and_verify_citations(
    answer: str,
    sources: List[VectorStoreResult]
) -> List[Citation]:
    """
    Extracts citations matching patterns like [doc_name.pdf, Page X] or [doc_name, Page X]
    and verifies whether the cited document and page number actually exist in the retrieved sources.
    """
    # Regex to find citations e.g. [paper.pdf, Page 3] or [paper.pdf, Page 3, Chunk 1] or [paper.pdf, p. 3]
    pattern = r'\[([^,\[\]]+),\s*(?:Page|page|p\.)\s*(\d+)(?:[^\]]*)?\]'
    matches = re.findall(pattern, answer)

    # Build lookup of valid (doc_name, page_number)
    valid_sources = {
        (s.doc_name.lower().strip(), int(s.page_number))
        for s in sources
    }
    # Also index by prefix without extension
    valid_names_no_ext = {
        (s.doc_name.rsplit(".", 1)[0].lower().strip(), int(s.page_number))
        for s in sources
    }

    citations: List[Citation] = []
    seen = set()

    for doc_str, page_str in matches:
        doc_clean = doc_str.strip()
        page_num = int(page_str)
        key = (doc_clean.lower(), page_num)
        if key in seen:
            continue
        seen.add(key)

        is_verified = (
            (doc_clean.lower(), page_num) in valid_sources or
            (doc_clean.rsplit(".", 1)[0].lower(), page_num) in valid_names_no_ext
        )

        citations.append(Citation(
            doc_name=doc_clean,
            page_number=page_num,
            raw_citation=f"[{doc_clean}, Page {page_num}]",
            is_verified=is_verified
        ))

    return citations


def call_ollama_llm(
    prompt: str,
    system_prompt: str,
    model: str = settings.LLM_MODEL,
    host: str = settings.OLLAMA_HOST,
    timeout: float = 60.0
) -> Optional[str]:
    """
    Calls Ollama REST API /api/generate synchronously.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for factual precision
            "top_p": 0.9
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
    except Exception:
        return None

    return None


def generate_extractive_fallback_answer(
    query: str,
    sources: List[VectorStoreResult]
) -> str:
    """
    High-fidelity deterministic synthesis used when Ollama is offline or in test environments.
    Extracts the most relevant sentences directly from top-scoring passages and appends verifiable citations.
    """
    if not sources:
        return INSUFFICIENT_EVIDENCE_PHRASE

    # Check top similarity score
    best_score = max(s.similarity_score for s in sources)
    if best_score < settings.SIMILARITY_THRESHOLD:
        return INSUFFICIENT_EVIDENCE_PHRASE

    query_terms = set(re.findall(r'\w+', query.lower())) - {"what", "is", "the", "are", "of", "and", "in", "to", "for"}
    
    selected_points = []
    for s in sources[:3]:
        # Split passage into sentences
        sentences = re.split(r'(?<=[.!?])\s+', s.text)
        best_sentence = ""
        best_overlap = -1
        for sent in sentences:
            sent_terms = set(re.findall(r'\w+', sent.lower()))
            overlap = len(query_terms.intersection(sent_terms))
            if overlap > best_overlap and len(sent.strip()) > 20:
                best_overlap = overlap
                best_sentence = sent.strip()

        if best_sentence:
            selected_points.append(f"- {best_sentence} [{s.doc_name}, Page {s.page_number}]")
        else:
            # Use snippet if no specific sentence matched
            snip = s.text[:140].replace("\n", " ").strip()
            selected_points.append(f"- {snip}... [{s.doc_name}, Page {s.page_number}]")

    if not selected_points:
        return INSUFFICIENT_EVIDENCE_PHRASE

    return (
        f"Based on the retrieved research evidence, here are the key findings:\n\n"
        + "\n\n".join(selected_points)
    )


class RAGPipeline:
    """
    Complete RAG Generation Pipeline for answering questions with verified source grounding.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.LLM_MODEL

    def answer_query(
        self,
        query: str,
        sources: List[VectorStoreResult],
        force_fallback: bool = False
    ) -> RAGResponse:
        start_time = time.time()

        # Check if sources are empty or scores are too low
        if not sources or len(sources) == 0:
            return RAGResponse(
                query=query,
                answer=INSUFFICIENT_EVIDENCE_PHRASE,
                sources=[],
                citations=[],
                is_insufficient_evidence=True,
                model_used="None (Early Refusal)",
                latency_seconds=round(time.time() - start_time, 3)
            )

        # Check top similarity against threshold
        top_score = max(s.similarity_score for s in sources)
        if top_score < settings.SIMILARITY_THRESHOLD:
            return RAGResponse(
                query=query,
                answer=INSUFFICIENT_EVIDENCE_PHRASE,
                sources=sources,
                citations=[],
                is_insufficient_evidence=True,
                model_used="None (Low Similarity Score)",
                latency_seconds=round(time.time() - start_time, 3)
            )

        formatted_context = format_context_for_prompt(sources)
        prompt = USER_PROMPT_TEMPLATE.format(context=formatted_context, question=query)

        raw_answer: Optional[str] = None
        used_model = self.model_name

        if not force_fallback:
            raw_answer = call_ollama_llm(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=self.model_name
            )

        if not raw_answer:
            # Fallback to local extractive grounded synthesis
            raw_answer = generate_extractive_fallback_answer(query, sources)
            used_model = f"Local Grounded Extractor (Fallback)"

        # Check if LLM explicitly said insufficient evidence
        is_insufficient = (
            INSUFFICIENT_EVIDENCE_PHRASE.lower() in raw_answer.lower() or
            "insufficient evidence" in raw_answer.lower()
        )

        citations = extract_and_verify_citations(raw_answer, sources)
        latency = round(time.time() - start_time, 3)

        return RAGResponse(
            query=query,
            answer=raw_answer,
            sources=sources,
            citations=citations,
            is_insufficient_evidence=is_insufficient,
            model_used=used_model,
            latency_seconds=latency
        )
