"""
ResearchPilot: Free, Local-First AI Research & Knowledge Agent.
Interactive Streamlit Web Application.
"""

import os
import io
import time
import json
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import settings
from src.ingestion import extract_pdf_pages, IngestionResult, clean_extracted_text
from src.chunking import (
    chunk_document_pages,
    chunk_ingestion_results,
    get_chunk_statistics,
    DocumentChunk
)
from src.vector_store import ResearchVectorStore, VectorStoreResult
from src.rag_pipeline import RAGPipeline, RAGResponse, INSUFFICIENT_EVIDENCE_PHRASE
from src.agent import ResearchAgent
from src.evaluation import EvaluationRunner, BenchmarkReport, get_system_hardware_info

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="ResearchPilot — Local AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Modern Glassmorphic Container Styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .citation-badge {
        display: inline-block;
        background-color: #0369a1;
        color: #f0f9ff;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 4px;
    }
    .citation-unverified {
        background-color: #b91c1c;
        color: #fef2f2;
    }
    .source-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 3px solid #38bdf8;
        padding: 12px 14px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .banner-box {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Session State Initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = ResearchVectorStore()
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
if "agent" not in st.session_state:
    st.session_state.agent = ResearchAgent(
        vector_store=st.session_state.vector_store,
        rag_pipeline=st.session_state.rag_pipeline
    )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs_meta" not in st.session_state:
    st.session_state.uploaded_docs_meta = {}
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None


# --- Sidebar Navigation & Model Status ---
with st.sidebar:
    st.title("🔬 ResearchPilot")
    st.caption("Free, Local-First AI Research Agent")

    st.markdown("---")
    st.subheader("System Status")
    
    ollama_info = settings.check_ollama_connectivity()
    if ollama_info["online"]:
        st.success(f"Ollama Active: {ollama_info['host']}")
        st.caption(f"LLM: `{settings.LLM_MODEL}` | Embed: `{settings.EMBEDDING_MODEL}`")
    else:
        st.warning("Ollama Offline")
        st.caption("Running in High-Fidelity Local Fallback Mode.")

    st.markdown("---")
    st.subheader("Indexing Overview")
    stats = st.session_state.vector_store.get_collection_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Indexed Chunks", stats.get("total_chunks", 0))
    with c2:
        st.metric("Documents", len(stats.get("distinct_documents", [])))

    if st.button("Clear Vector Index", use_container_width=True, type="secondary"):
        st.session_state.vector_store.clear_collection()
        st.session_state.uploaded_docs_meta = {}
        st.session_state.chat_history = []
        st.success("Vector store cleared.")
        st.rerun()

    st.markdown("---")
    enable_web = st.checkbox("Enable Optional Web Research", value=False, help="Adds external web citations tagged separately from uploaded PDF documents.")


# --- Main Tab Navigation ---
tab_chat, tab_docs, tab_compare, tab_report, tab_eval = st.tabs([
    "💬 Research Chat",
    "📑 Document Library",
    "⚖️ Multi-Doc Synthesis",
    "📄 Report Dossier",
    "📊 Evaluation Dashboard"
])


# =========================================================================
# TAB 1: RESEARCH CHAT & ASSISTANT
# =========================================================================
with tab_chat:
    st.markdown("""
    <div class="banner-box">
        <h3 style="margin: 0; color: #f8fafc;">AI Research Knowledge Assistant</h3>
        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
            Ask questions grounded strictly in your uploaded PDF papers. Citations and evidence snippets are verified automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF whitepapers, preprints, or reports."
    )

    if uploaded_files:
        new_uploads = [f for f in uploaded_files if f.name not in st.session_state.uploaded_docs_meta]
        if new_uploads:
            with st.status(f"Ingesting {len(new_uploads)} document(s)...", expanded=True) as status:
                for file_obj in new_uploads:
                    st.write(f"Extracting text from `{file_obj.name}`...")
                    bytes_data = file_obj.read()
                    ingest_res = extract_pdf_pages(bytes_data, file_obj.name)

                    if not ingest_res.is_success:
                        st.error(f"Error in `{file_obj.name}`: {', '.join(ingest_res.errors)}")
                        continue

                    if ingest_res.warnings:
                        for w in ingest_res.warnings:
                            st.warning(f"`{file_obj.name}`: {w}")

                    # Chunking
                    chunks = chunk_document_pages(ingest_res.pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                    st.write(f"Indexing {len(chunks)} chunks into Chroma vector store...")
                    st.session_state.vector_store.add_chunks(chunks)

                    st.session_state.uploaded_docs_meta[file_obj.name] = {
                        "pages": ingest_res.total_pages,
                        "valid_pages": ingest_res.valid_pages,
                        "chunks": len(chunks),
                        "words": ingest_res.total_words,
                        "has_text": ingest_res.has_text
                    }

                status.update(label="Document ingestion & vector indexing complete!", state="complete", expanded=False)
            st.rerun()

    # Sample Quick Questions
    st.markdown("##### Quick Research Prompts")
    q_col1, q_col2, q_col3 = st.columns(3)
    preset_q = None
    with q_col1:
        if st.button("🔍 Main Findings & Methods", use_container_width=True):
            preset_q = "What are the main findings and methodologies proposed in these documents? Include source citations."
    with q_col2:
        if st.button("⚖️ Compare Approaches", use_container_width=True):
            preset_q = "Compare the approaches used across the uploaded papers, highlight divergences, and cite supporting documents."
    with q_col3:
        if st.button("⚠️ Reported Limitations", use_container_width=True):
            preset_q = "What limitations, assumptions, or dataset constraints do the authors explicitly mention?"

    # Chat Display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                cit_html = "<div><strong>Citations:</strong> "
                for c in msg["citations"]:
                    cls = "citation-badge" if c.get("is_verified") else "citation-badge citation-unverified"
                    v_label = " (Verified)" if c.get("is_verified") else " (Unverified)"
                    cit_html += f"<span class='{cls}'>{c.get('raw_citation', '')}{v_label}</span>"
                cit_html += "</div>"
                st.markdown(cit_html, unsafe_allow_html=True)

            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Supporting Evidence Passages ({len(msg['sources'])})"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source #{idx}:</strong> <code>{src.get('doc_name')}</code> — Page {src.get('page_number')} | <em>Score: {src.get('similarity_score')}</em><br/>
                            <p style="margin-top:6px; color:#cbd5e1;">{src.get('text')}</p>
                        </div>
                        """, unsafe_allow_html=True)

            if "web_sources" in msg and msg["web_sources"]:
                with st.expander(f"🌐 External Web Sources ({len(msg['web_sources'])})"):
                    for w in msg["web_sources"]:
                        st.markdown(f"- [{w.get('title')}]({w.get('url')}) — *Retrieved: {w.get('retrieval_date')}*")

    # Chat Input
    user_query = st.chat_input("Ask a research question across your documents...")
    effective_query = preset_q or user_query

    if effective_query:
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": effective_query})
        with st.chat_message("user"):
            st.markdown(effective_query)

        # Assistant Processing
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document evidence with Research Agent..."):
                agent_res = st.session_state.agent.run(effective_query, enable_web_search=enable_web)
                answer_text = agent_res.get("generation", "")
                chunks = agent_res.get("retrieved_chunks", [])
                citations = agent_res.get("citations", [])
                web_res = agent_res.get("web_sources", [])

                st.markdown(answer_text)

                if citations:
                    cit_html = "<div><strong>Citations:</strong> "
                    for c in citations:
                        cls = "citation-badge" if c.get("is_verified") else "citation-badge citation-unverified"
                        v_label = " (Verified)" if c.get("is_verified") else " (Unverified)"
                        cit_html += f"<span class='{cls}'>{c.get('raw_citation', '')}{v_label}</span>"
                    cit_html += "</div>"
                    st.markdown(cit_html, unsafe_allow_html=True)

                if chunks:
                    with st.expander(f"📚 Supporting Evidence Passages ({len(chunks)})"):
                        for idx, src in enumerate(chunks, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source #{idx}:</strong> <code>{src.doc_name}</code> — Page {src.page_number} | <em>Score: {src.similarity_score}</em><br/>
                                <p style="margin-top:6px; color:#cbd5e1;">{src.text}</p>
                            </div>
                            """, unsafe_allow_html=True)

                if web_res:
                    with st.expander(f"🌐 External Web Sources ({len(web_res)})"):
                        for w in web_res:
                            st.markdown(f"- [{w.get('title')}]({w.get('url')}) — *Retrieved: {w.get('retrieval_date')}*")

                st.caption(f"⏱️ Retrieval & Inference latency: {agent_res.get('latency_seconds', 0)}s | Evidence score: {agent_res.get('sufficiency_score', 0)}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "citations": citations,
                    "sources": [
                        {
                            "doc_name": s.doc_name,
                            "page_number": s.page_number,
                            "similarity_score": s.similarity_score,
                            "text": s.text
                        } for s in chunks
                    ],
                    "web_sources": web_res
                })


# =========================================================================
# TAB 2: DOCUMENT LIBRARY & CHUNKS EXPLORER
# =========================================================================
with tab_docs:
    st.subheader("Indexed Document Inventory")

    if not st.session_state.uploaded_docs_meta:
        st.info("No documents uploaded yet. Use the Research Chat tab to upload PDF papers.")
    else:
        doc_rows = []
        for name, m in st.session_state.uploaded_docs_meta.items():
            doc_rows.append({
                "Document Name": name,
                "Total Pages": m.get("pages", 0),
                "Valid Pages": m.get("valid_pages", 0),
                "Total Chunks": m.get("chunks", 0),
                "Total Words": m.get("words", 0),
                "Text Extracted": "✅ Yes" if m.get("has_text") else "⚠️ Scanned / Empty"
            })
        st.dataframe(doc_rows, use_container_width=True)

        st.markdown("---")
        st.subheader("Chunk Inspector")
        selected_doc = st.selectbox("Select document to inspect chunks:", list(st.session_state.uploaded_docs_meta.keys()))

        if selected_doc:
            # Query chunks for this doc
            matching_chunks = st.session_state.vector_store.similarity_search("findings methodology research", top_k=20, filter_doc=selected_doc)
            if matching_chunks:
                st.caption(f"Showing top {len(matching_chunks)} chunks for `{selected_doc}`:")
                for c in matching_chunks:
                    with st.expander(f"📄 Page {c.page_number} — Chunk ID: `{c.chunk_id}`"):
                        st.text(c.text)
                        st.caption(f"Offsets: {c.metadata.get('char_start')} to {c.metadata.get('char_end')} | Score: {c.similarity_score}")
            else:
                st.write("No chunks found for this document.")


# =========================================================================
# TAB 3: MULTI-DOCUMENT COMPARISON & SYNTHESIS
# =========================================================================
with tab_compare:
    st.subheader("Cross-Document Comparative Synthesis")
    st.caption("Synthesize and contrast methodologies, empirical findings, and limitations across papers.")

    doc_names = list(st.session_state.uploaded_docs_meta.keys())
    if len(doc_names) < 2:
        st.warning("Please upload at least 2 PDF documents to perform cross-document comparison.")
    else:
        selected_to_compare = st.multiselect("Select documents for comparison:", doc_names, default=doc_names[:min(3, len(doc_names))])
        compare_topic = st.text_input("Comparison Focus Area:", "Core methodology, main findings, and acknowledged limitations")

        if st.button("Generate Comparative Synthesis Matrix", type="primary"):
            with st.spinner("Synthesizing evidence across selected documents..."):
                all_evidence = []
                for d in selected_to_compare:
                    d_chunks = st.session_state.vector_store.similarity_search(compare_topic, top_k=3, filter_doc=d)
                    all_evidence.extend(d_chunks)

                prompt = f"Compare and contrast the following documents regarding '{compare_topic}'. Present the synthesis in a clear comparative table followed by key divergences:\n"
                rag_resp = st.session_state.rag_pipeline.answer_query(prompt, all_evidence)

                st.markdown("### Comparative Results")
                st.markdown(rag_resp.answer)

                if rag_resp.citations:
                    st.markdown("#### Document Citations")
                    for c in rag_resp.citations:
                        st.markdown(f"- `{c.raw_citation}` {'✅ Verified' if c.is_verified else '⚠️ Unverified'}")


# =========================================================================
# TAB 4: RESEARCH REPORT GENERATOR
# =========================================================================
with tab_report:
    st.subheader("Downloadable Markdown Research Report")
    st.caption("Generate a structured research report summarizing all analyzed literature.")

    report_title = st.text_input("Report Title:", "AI Research Synthesis & Literature Review")
    author_name = st.text_input("Prepared By:", "ResearchPilot Agent")
    research_scope = st.text_area("Research Inquiries / Subtopics to Cover:", "1. Overview of Primary Methodologies\n2. Key Experimental Findings\n3. Limitations & Future Directions")

    if st.button("Generate Complete Dossier", type="primary"):
        with st.spinner("Compiling multi-document research dossier..."):
            lines = [
                f"# {report_title}",
                f"",
                f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
                f"**Author / Agent:** {author_name}  ",
                f"**Analyzed Documents:** {', '.join(st.session_state.uploaded_docs_meta.keys()) if st.session_state.uploaded_docs_meta else 'Benchmark Library'}  ",
                f"",
                f"## Executive Summary",
                f"This report consolidates evidence retrieved across uploaded literature, structured with verifiable page citations.",
                f"",
                f"## Research Inquiries & Findings",
                f""
            ]

            inquiries = [i.strip() for i in research_scope.split("\n") if i.strip()]
            for inq in inquiries:
                lines.append(f"### {inq}")
                retrieved = st.session_state.vector_store.similarity_search(inq, top_k=3)
                ans = st.session_state.rag_pipeline.answer_query(inq, retrieved)
                lines.append(ans.answer)
                lines.append("")
                if ans.citations:
                    lines.append("**Citations:**")
                    for c in ans.citations:
                        lines.append(f"- {c.raw_citation}")
                    lines.append("")

            lines.append("## Provenance & Grounding Verification")
            lines.append("All statements above are mapped to specific source pages in the local ChromaDB index.")

            full_report = "\n".join(lines)
            st.markdown(full_report)

            st.download_button(
                label="📥 Download Research Report (.md)",
                data=full_report,
                file_name=f"research_report_{int(time.time())}.md",
                mime="text/markdown"
            )


# =========================================================================
# TAB 5: EVALUATION DASHBOARD
# =========================================================================
with tab_eval:
    st.subheader("Evaluation & Benchmarking Dashboard")
    st.markdown("""
    Evaluate retrieval precision, answer faithfulness, citation correctness, and refusal accuracy against the ground-truth benchmark suite described in the README.
    """)

    e_col1, e_col2 = st.columns([1, 2])
    with e_col1:
        if st.button("🚀 Run Automated Evaluation Benchmark", type="primary", use_container_width=True):
            with st.spinner("Running 4-query evaluation benchmark suite..."):
                runner = EvaluationRunner()
                st.session_state.eval_report = runner.run_evaluation(top_k=2)
                st.success("Evaluation suite completed!")

    if st.session_state.eval_report:
        rep: BenchmarkReport = st.session_state.eval_report

        # Metric Tiles
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Recall@k / Hit Rate</div>
                <div class="metric-val">{rep.recall_at_k * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Faithfulness Score</div>
                <div class="metric-val">{rep.faithfulness_avg * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Citation Correctness</div>
                <div class="metric-val">{rep.citation_correctness_avg * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Refusal Accuracy</div>
                <div class="metric-val">{rep.refusal_accuracy * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        l1, l2 = st.columns(2)
        with l1:
            st.info(f"⚡ **Avg Retrieval Latency:** {rep.avg_retrieval_latency_ms:.1f} ms")
        with l2:
            st.info(f"⚡ **Avg Generation Latency:** {rep.avg_generation_latency_ms:.1f} ms")

        st.markdown("### Benchmark Item Breakdown")
        table_rows = []
        for it in rep.items:
            table_rows.append({
                "Query": it.query,
                "Answerable": "Yes" if it.is_answerable else "No (Negative Test)",
                "Retrieval Hit": "✅ Yes" if it.retrieval_hit else "❌ Miss",
                "Refusal Correct": "✅ Yes" if it.refusal_correct else "❌ Failed",
                "Faithfulness": f"{it.faithfulness_score * 100:.0f}%",
                "Citation Correct": f"{it.citation_correctness_rate * 100:.0f}%",
                "Latency (ms)": f"{it.retrieval_latency_ms + it.generation_latency_ms:.0f}"
            })
        st.dataframe(table_rows, use_container_width=True)

        st.download_button(
            label="📥 Export Benchmark Report (.md)",
            data=rep.to_markdown(),
            file_name="evaluation_benchmark_report.md",
            mime="text/markdown"
        )
