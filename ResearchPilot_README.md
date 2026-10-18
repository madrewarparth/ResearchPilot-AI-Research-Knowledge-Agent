# ResearchPilot --- Free, Local-First AI Research & Knowledge Agent

ResearchPilot is a portfolio project for building an AI research
assistant that answers questions across uploaded documents, retrieves
supporting evidence, and produces structured summaries with citations.

The project is designed to run **without paid APIs or subscriptions**.
It uses a local LLM through Ollama, local embedding models, and
open-source Python libraries. A computer with enough memory is required
to run models locally.

> **Project status:** Portfolio project blueprint. Update this README as
> you implement and test each feature. Do not claim features or
> evaluation results that are not yet working in your repository.

## Table of Contents

-   [Why ResearchPilot?](#why-researchpilot)
-   [Features](#features)
-   [Free Technology Stack](#free-technology-stack)
-   [Architecture](#architecture)
-   [How RAG Works](#how-rag-works)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Run the Application](#run-the-application)
-   [Example Questions](#example-questions)
-   [Project Structure](#project-structure)
-   [Development Roadmap](#development-roadmap)
-   [Evaluation Plan](#evaluation-plan)
-   [Privacy and Cost](#privacy-and-cost)
-   [Troubleshooting](#troubleshooting)
-   [Limitations](#limitations)
-   [Contributing](#contributing)
-   [License](#license)

## Why ResearchPilot?

Researchers and students often need to read multiple documents, find
relevant evidence, compare claims, and prepare summaries. ResearchPilot
aims to make this process faster while keeping answers grounded in the
documents supplied by the user.

Example: upload several papers about renewable energy and ask:

> Compare the approaches used in these papers, summarize their
> limitations, and cite the documents that support each point.

ResearchPilot should retrieve relevant passages, generate an answer from
those passages, and show the source document and page or chunk reference
where available.

## Features

### Core MVP

-   Upload PDF documents.
-   Extract text and split it into manageable chunks.
-   Create local vector embeddings and store them in Chroma.
-   Retrieve relevant passages for a user question.
-   Generate answers using a locally running language model.
-   Display retrieved source snippets so users can verify the answer.
-   Provide a Streamlit interface for uploading documents and chatting.
-   Handle empty documents and questions with insufficient evidence.

### Planned enhancements

-   Multi-document comparison and structured research summaries.
-   LangGraph workflow that routes between document retrieval,
    clarification, and answer generation.
-   Optional web research tool, with URLs and dates shown separately
    from uploaded-document evidence.
-   Downloadable Markdown research reports.
-   Evaluation dashboard for retrieval quality, answer faithfulness,
    latency, and local resource usage.
-   Docker support and automated tests.

Only mark a feature as complete after implementing and testing it.

## Free Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Language                Python                  Application logic

  Local LLM runtime       Ollama                  Runs an open model on
                                                  your computer

  Starter chat model      `llama3.2:3b`           Local answer generation

  Embeddings              `nomic-embed-text`      Converts text into
                          through Ollama          vectors

  Vector store            ChromaDB                Stores and searches
                                                  document chunks

  Document parsing        PyMuPDF                 Extracts text from PDFs

  UI                      Streamlit               Web-based local
                                                  interface

  RAG / agent             LangChain and LangGraph Retrieval chain and
  orchestration                                   later agent workflow

  Testing                 pytest                  Automated tests

  Version control         Git and GitHub          Source control and
                                                  portfolio hosting
  -----------------------------------------------------------------------

All listed software is available without a paid subscription. Model
downloads use disk space and bandwidth, and running local models uses
your computer's CPU/RAM or supported GPU. Model availability and
hardware requirements can change.

**No OpenAI API key is required for the local-first version.** Do not
add paid APIs as mandatory dependencies.

## Architecture

``` text
PDF files
   |
   v
PDF text extraction (PyMuPDF)
   |
   v
Text cleaning and chunking
   |
   v
Local embeddings (Ollama: nomic-embed-text)
   |
   v
Chroma vector database
   |
User question
   |
   v
Question embedding + similarity search
   |
   v
Relevant passages and source metadata
   |
   v
Local LLM (Ollama: llama3.2:3b)
   |
   v
Answer with source references in Streamlit
```

### Planned agent workflow

After the basic RAG flow works, LangGraph can coordinate a controlled
workflow:

1.  Inspect the user's question.
2.  Retrieve relevant document passages.
3.  Check whether the retrieved evidence is sufficient.
4.  If evidence is insufficient, ask for clarification or report the
    limitation.
5.  Generate an answer grounded in retrieved evidence.
6.  Return the answer with source references.

Add optional web search only after the document-only workflow works
reliably. Keep web sources clearly distinguished from uploaded
documents.

## How RAG Works

Retrieval-Augmented Generation (RAG) combines information retrieval with
language-model generation:

1.  **Ingestion:** Extract text from each PDF.
2.  **Chunking:** Split text into smaller overlapping passages.
3.  **Embedding:** Convert each passage into a numerical vector.
4.  **Indexing:** Store vectors and source metadata in Chroma.
5.  **Retrieval:** Embed the question and retrieve the most relevant
    passages.
6.  **Generation:** Provide those passages to the local LLM as context.
7.  **Grounding:** Show source references and avoid presenting
    unsupported claims as facts.

RAG can reduce unsupported answers, but it does not eliminate
hallucinations. Users should verify important claims against the
original documents.

## Prerequisites

-   Python 3.10 or 3.11 recommended.
-   Git.
-   Ollama installed and running.
-   Enough disk space for the selected model files.
-   A computer with sufficient RAM for the chosen model. Smaller models
    are generally easier to run on modest hardware.

Official installation references:

-   Ollama: https://ollama.com/download
-   Python: https://www.python.org/downloads/
-   Git: https://git-scm.com/downloads

## Installation

### 1. Download or clone the repository

``` bash
git clone https://github.com/<your-username>/ResearchPilot.git
cd ResearchPilot
```

Replace `<your-username>` with your GitHub username and update the
repository name if you choose another one.

### 2. Create a virtual environment

**Windows PowerShell:**

``` powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, you can use Command Prompt instead:

``` bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Ollama models

Install Ollama first, then open a terminal:

``` bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Start the Ollama application/service. Verify that the models are
available:

``` bash
ollama list
```

If the chat model is too slow or does not fit your hardware, choose a
smaller compatible model available in Ollama and update the model name
in your application configuration.

### 4. Install Python dependencies

Create a `requirements.txt` file containing the packages needed for the
MVP:

``` text
streamlit
pymupdf
chromadb
ollama
langchain
langchain-community
langchain-text-splitters
langchain-ollama
langgraph
python-dotenv
pytest
```

Then install them:

``` bash
pip install -r requirements.txt
```

For a reproducible portfolio submission, test the dependency set in a
clean virtual environment and pin tested versions before publishing.

## Run the Application

Once the implementation is in place, start the Streamlit application
from the project root:

``` bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`.

The README describes the intended workflow; this command will work only
after you have created `app.py` and implemented the features.

## Example Questions

Use questions that can be answered from the uploaded documents:

-   What are the main findings of this paper? Include source references.
-   Compare the methods used in Document A and Document B.
-   What limitations do the authors mention?
-   Which documents support the claim that solar efficiency improved?
-   What information is missing from the uploaded documents to answer
    this question?
-   Summarize the key findings in a table and cite each finding.

A good system should say when the documents do not contain enough
evidence instead of inventing an answer.

## Project Structure

A suggested structure for the completed repository:

``` text
ResearchPilot/
├── app.py                    # Streamlit user interface
├── config.py                 # Model and application settings
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── ingestion.py          # PDF extraction and validation
│   ├── chunking.py           # Text splitting
│   ├── vector_store.py       # Chroma indexing and retrieval
│   ├── rag_pipeline.py       # Prompting and answer generation
│   ├── agent.py              # Later: LangGraph workflow
│   └── evaluation.py         # Evaluation helpers
├── tests/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   └── test_rag_pipeline.py
├── data/
│   └── .gitkeep              # Keep private documents out of Git
└── chroma_db/                # Local index; do not commit generated data
```

You can start with a single `app.py` and split it into modules as the
project grows.

## Development Roadmap

### Phase 1 --- Document ingestion

-   [ ] Upload one or more PDFs.
-   [ ] Extract text and handle scanned/image-only PDFs gracefully.
-   [ ] Store document names and page metadata.
-   [ ] Validate empty, encrypted, or unsupported files.

### Phase 2 --- Retrieval and answers

-   [ ] Split text into overlapping chunks.
-   [ ] Generate local embeddings.
-   [ ] Index chunks in Chroma.
-   [ ] Retrieve top relevant passages.
-   [ ] Generate answers using Ollama.
-   [ ] Display citations and source snippets.
-   [ ] Add a clear "insufficient evidence" response.

### Phase 3 --- Agent workflow

-   [ ] Build a LangGraph state and nodes.
-   [ ] Route questions through retrieval and answer generation.
-   [ ] Add evidence sufficiency checks.
-   [ ] Add a controlled retry or clarification path.
-   [ ] Add optional web search as a separate, clearly cited tool.

### Phase 4 --- Evaluation and polish

-   [ ] Create a small set of questions with known supporting passages.
-   [ ] Measure retrieval hit rate or Recall@k.
-   [ ] Review answer faithfulness and citation correctness.
-   [ ] Measure latency and record model/hardware details.
-   [ ] Add automated tests and error handling.
-   [ ] Capture screenshots and a short demo video.
-   [ ] Document limitations and publish a clean GitHub repository.

## Evaluation Plan

Do not report invented accuracy or performance numbers. Evaluate the
implementation on a small, documented test set.

  -----------------------------------------------------------------------
  Metric                              What it checks
  ----------------------------------- -----------------------------------
  Recall@k / Hit Rate@k               Whether relevant passages appear in
                                      the retrieved top-k results

  Faithfulness                        Whether generated claims are
                                      supported by the supplied evidence

  Citation correctness                Whether cited sources actually
                                      support the associated claims

  Answer relevance                    Whether the answer addresses the
                                      question

  Latency                             Time taken for ingestion,
                                      retrieval, and generation

  Resource usage                      Approximate RAM, disk, and model
                                      requirements on the test machine
  -----------------------------------------------------------------------

Record the test dataset, model names, settings, hardware, and date.
Manual review is especially important for citations and unsupported
claims. An evaluation framework can help, but its scores should not be
treated as proof of correctness.

## Privacy and Cost

-   **API cost:** The core version is designed to use local models
    without paid API calls.
-   **Subscriptions:** No paid subscription is required for the core
    workflow.
-   **Internet:** Internet is needed to install packages and download
    models. The local document-question-answering flow can then run
    without an external LLM API.
-   **Privacy:** Uploaded documents and the vector index can remain on
    your computer if you do not add cloud integrations. Review all
    dependencies and optional tools before using confidential files.
-   **Hardware:** Local inference is not cost-free in terms of
    electricity, memory, storage, or processing time.
-   **Web search:** Optional web search may depend on third-party
    services, their current availability, and their usage terms. It is
    not required for the MVP.

Never commit API keys, private PDFs, personal data, model files, or
local vector database contents to GitHub.

## Troubleshooting

### Ollama is not running

Start the Ollama application/service and check:

``` bash
ollama list
```

If the model cannot be found, pull it using the exact model name
configured in the application.

### Model responses are slow

Try a smaller local model, reduce the amount of retrieved context, and
avoid processing too many documents at once. Performance depends on your
hardware.

### No text is extracted from a PDF

Some PDFs are scans containing images instead of selectable text. The
MVP needs an OCR feature to support these files; OCR is not
automatically provided by basic text extraction.

### Answers do not contain correct citations

Preserve document name, page number, and chunk ID as metadata throughout
ingestion and retrieval. Display only citations that map to retrieved
source chunks, and test citation correctness.

### Dependency conflicts

Use the recommended Python version in a clean virtual environment. Once
the project runs, pin the versions that you have actually tested.

## Limitations

-   Local models may be less capable than larger hosted models.
-   PDF text extraction can fail on scans, complex tables, and unusual
    layouts.
-   Retrieval can miss relevant passages.
-   Language models may generate incorrect or unsupported claims.
-   Web search, if added, may be unavailable or subject to service
    restrictions.
-   The application is a research aid, not a substitute for checking
    original sources.

## Contributing

Contributions and suggestions are welcome. Open an issue describing the
problem or proposed feature before submitting a major change.

1.  Fork the repository.
2.  Create a feature branch.
3.  Add or update tests.
4.  Run the tests locally.
5.  Submit a pull request describing the change and its limitations.

## License

Choose a license before publishing the repository. The MIT License is a
common option for open-source portfolio projects; add a `LICENSE` file
only after deciding which license you want to use. Also check the
licenses of any models and datasets you distribute.
