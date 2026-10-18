# ResearchPilot — Free, Local-First AI Research & Knowledge Agent

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI%20Framework-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain%20%2F%20LangGraph-Orchestration-1C3C3C)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ResearchPilot is an open-source, local-first AI research assistant that answers questions across uploaded research papers and PDF documents, retrieves verifiable supporting evidence, and produces structured research summaries with exact citations.

The project runs **100% locally without paid APIs, subscriptions, or external data leaks**. It is powered by local LLMs through Ollama (`llama3.2:3b`), local embedding models (`nomic-embed-text`), ChromaDB vector store, and a LangGraph-orchestrated evidence-sufficiency workflow.

---

## Table of Contents

- [Why ResearchPilot?](#why-researchpilot)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [LangGraph Agent Workflow](#langgraph-agent-workflow)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Quickstart](#installation--quickstart)
- [Running the Application](#running-the-application)
- [Example Queries](#example-queries)
- [Development Roadmap Status](#development-roadmap-status)
- [Evaluation Plan & Metrics](#evaluation-plan--metrics)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contributing](#contributing)
- [License](#license)

---

## Why ResearchPilot?

Researchers, students, and engineers frequently read multi-page whitepapers, compare empirical results, and synthesize literature. ResearchPilot streamlines this workflow while ensuring:
1. **Strict Source Grounding:** Every factual assertion is mapped to an exact document name and page number citation (`[Doc, Page X]`).
2. **Honest "Insufficient Evidence" Detection:** If the uploaded documents do not contain the answer, ResearchPilot clearly states the evidence gap instead of hallucinating.
3. **Complete Privacy:** Uploaded documents, vector embeddings, and LLM inferences remain strictly on your local machine.

---

## Key Features

### Core MVP
- 📄 **Multi-PDF Ingestion & Validation:** Extract text, preserve page numbers, detect scanned/empty/encrypted PDFs gracefully.
- ✂️ **Deterministic Overlapping Chunking:** Configurable chunk sizes (default 800 chars) with 150-char overlap, keeping sentence context and full chunk metadata.
- 🧠 **Local Vector Search:** Indexed in ChromaDB using `nomic-embed-text` with fast cosine similarity retrieval.
- 💬 **Grounded Local LLM Generation:** Prompted Ollama `llama3.2:3b` with strict citation mandates and refusal of unsupported speculation.
- 🔍 **Interactive Evidence Viewer:** Expandable source snippets showing chunk text, page numbers, and similarity metrics.
- 🖥️ **Streamlit Web Application:** Modern, responsive UI with multi-document upload, live Q&A, and interactive configuration.

### Advanced Enhancements
- 🕸️ **LangGraph Agentic Routing:** Question inspection, retrieval, evidence sufficiency grading, query rewriting, and fallback clarification.
- 📊 **Multi-Document Comparison Matrix:** Cross-document synthesis table comparing findings, methodologies, and limitations.
- 📝 **Markdown Research Report Export:** One-click generation of structured research dossiers with executive summaries and citation appendices.
- 📈 **Evaluation & Benchmarking Suite:** Automated measurement of Recall@k, answer faithfulness, citation correctness, and inference latency.
- 🐳 **Docker & Docker-Compose Support:** Reproducible containerization with automated volume mounting.

---

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10 / 3.11 | Core application logic and pipeline |
| **Local LLM Runtime** | Ollama | Runs local open-weights language models |
| **Default Chat Model** | `llama3.2:3b` | Local answer generation & synthesis |
| **Embedding Model** | `nomic-embed-text` | High-accuracy local text embeddings |
| **Vector Database** | ChromaDB | Local vector store with persistent storage |
| **PDF Extraction** | PyMuPDF (`fitz`) / `pypdf` | High-fidelity text extraction & metadata tracking |
| **Orchestration** | LangChain & LangGraph | Agentic flow, sufficiency grading & query routing |
| **User Interface** | Streamlit | Reactive web dashboard with modern UI |
| **Testing** | pytest | Comprehensive automated unit & integration tests |
| **Containerization**| Docker & Docker-Compose | Containerized deployment |

---

## System Architecture

```text
Uploaded PDF Files
       │
       ▼
PDF Text Extraction (PyMuPDF / pypdf)
       │  (Extract text, validate non-empty, record page numbers)
       ▼
Text Cleaning & Overlapping Chunking (800 chars, 150 overlap)
       │  (Attach metadata: doc_id, file_name, page_number, chunk_id)
       ▼
Local Embeddings (Ollama: nomic-embed-text)
       │
       ▼
Chroma Vector Database (Local Persistence)
       │
User Question
       │
       ▼
LangGraph Agent Workflow:
 ┌─────────────────────────────────────────────────────────┐
 │ 1. Inspect Question & Extract Keywords                  │
 │ 2. Retrieve Top-k Semantic Passages                     │
 │ 3. Grade Evidence Sufficiency                           │
 │    ├─► Insufficient ──► Query Rewrite Retry ──► Fallback│
 │    └─► Sufficient   ──► Grounded Generation             │
 │ 4. Verify Inline Citations against Retrieved Sources    │
 └─────────────────────────────────────────────────────────┘
       │
       ▼
Formatted Answer with Interactive Citations in Streamlit
```

---

## Project Structure

```text
ResearchPilot/
├── app.py                      # Streamlit interactive application
├── config.py                   # Centralized application settings & connectivity
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & roadmap
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── .env.example                # Sample environment variables
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Docker Compose orchestration
├── pytest.ini                  # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── ingestion.py            # PDF parsing, validation & metadata extraction
│   ├── chunking.py             # Overlapping text chunker & metadata preservation
│   ├── vector_store.py         # Chroma indexing, retrieval & local embeddings
│   ├── rag_pipeline.py         # Prompt engineering, LLM generation & citation parser
│   ├── agent.py                # LangGraph agent workflow with sufficiency grading
│   └── evaluation.py           # Recall@k, faithfulness & citation correctness metrics
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py       # PDF validation and extraction tests
│   ├── test_chunking.py        # Chunk size, overlap and boundary tests
│   ├── test_vector_store.py    # Embedding and Chroma vector search tests
│   ├── test_rag_pipeline.py    # Grounding, citation extraction and refusal tests
│   ├── test_agent.py           # LangGraph workflow and routing tests
│   └── test_evaluation.py      # Benchmark evaluation metrics tests
├── data/
│   └── .gitkeep                # Gitkeep for private document storage
└── chroma_db/                  # Local vector database (ignored in git)
```

---

## Installation & Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/madrewarparth/ResearchPilot-AI-Research-Knowledge-Agent.git
cd ResearchPilot-AI-Research-Knowledge-Agent
```

### 2. Set Up a Virtual Environment

**Windows PowerShell:**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install and Run Ollama Models

Install [Ollama](https://ollama.com/), then pull the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify installed models:
```bash
ollama list
```

---

## Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

*Note: If Ollama is not running, ResearchPilot automatically detects this and offers an Offline Demonstration mode so you can still test document processing, chunking, and mock RAG evaluation seamlessly.*

---

## Example Queries

- *"What are the key methodologies proposed in this paper? Include source citations."*
- *"Compare the experimental results between Document A and Document B in a table."*
- *"What limitations do the authors acknowledge regarding their dataset?"*
- *"Which documents substantiate the claims regarding energy efficiency?"*
- *"What details are missing from the uploaded documents to answer this query?"*

---

## Development Roadmap Status

- [x] **Phase 1 — Document Ingestion:** Multi-PDF upload, text extraction, page metadata tracking, scan/empty/encryption validation.
- [x] **Phase 2 — Chunking & Retrieval:** Deterministic text splitting with boundary awareness, Chroma vector storage, local embeddings.
- [x] **Phase 2 — Grounded Generation:** Prompt construction with citation enforcement, insufficient evidence detection, verified source snippets.
- [x] **Phase 3 — Agent Workflow:** LangGraph state machine, query routing, evidence sufficiency grading, fallback clarification.
- [x] **Phase 4 — Multi-Doc Synthesis & Reporting:** Cross-document comparison matrix, downloadable Markdown research reports.
- [x] **Phase 4 — Evaluation Dashboard:** Automated benchmark checking Recall@k, faithfulness, citation validity, and latency.
- [x] **Phase 4 — Containerization & Tests:** Full pytest test suite, Dockerfile, docker-compose configuration.

---

## Evaluation Plan & Metrics

| Metric | Description | Target |
| :--- | :--- | :--- |
| **Recall@k / Hit Rate@k** | Proportion of queries where ground-truth context is in top-k results | > 85% |
| **Faithfulness** | Percentage of claims directly supported by retrieved text chunks | > 90% |
| **Citation Correctness** | Verification that cited `[Doc, Page]` strictly matches evidence | > 95% |
| **Refusal Accuracy** | Correct refusal on unanswerable/out-of-scope queries | 100% |
| **Retrieval Latency** | ChromaDB query and cosine similarity computation | < 80ms |

Run the automated evaluation suite:
```bash
pytest tests/ -v
```

---

## Docker Deployment

To run ResearchPilot using Docker:

```bash
docker compose up --build
```

Access the Streamlit application at `http://localhost:8501`.

---

## License

This project is licensed under the [MIT License](LICENSE).
