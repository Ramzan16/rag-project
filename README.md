# RAG Research Assistant

A CLI-based Retrieval-Augmented Generation (RAG) system designed to fetch, ingest, and query research papers from arXiv using modern LLM providers and vector databases.

## 🚀 Features

- **arXiv Integration**: Search and download research papers directly using the arXiv API.
- **Batch Data Ingestion**: Robust pipeline for loading PDFs, cleaning text, chunking, and batch-uploading to a vector database.
- **Multi-Provider Support**: Switch seamlessly between local (Ollama) and cloud-based (Google Gemini) LLM providers.
- **Vector Database**: Integration with **Qdrant** for efficient similarity searches.
- **Structured Logging**: Comprehensive, level-based logging for tracking ingestion progress, retrieval times, and generation metrics.
- **Scalable Architecture**: Factory-based provider pattern for easy extension to new LLMs.

## 🏗️ Architecture

- **`config/`**: Centralized configuration management using `pydantic-settings`.
- **`llm_provider/`**: Abstraction layer for LLM interactions (Chat and Embeddings).
- **`services/`**:
  - `ArxivService`: Handles paper discovery and PDF downloads.
  - `IngestService`: Manages the document-to-vector pipeline.
  - `RagService`: Orchestrates the retrieval and generation logic.
- **`main.py`**: Unified CLI entry point.

## 🛠️ Setup

### Prerequisites

- **Python**: 3.12+
- **Vector DB**: A running [Qdrant](https://qdrant.tech/) instance (local or Cloud).
- **Package Manager**: [uv](https://github.com/astral-sh/uv) is recommended.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd rag-project
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    Create a `.env` file based on `.env.example`:
    ```bash
    cp .env.example .env
    ```
    Update the values in `.env` with your API keys and service URLs.

## 📖 Usage

The CLI supports three primary commands: `fetch`, `ingest`, and `rag`.

### 1. Fetch Research Papers
Search arXiv and download PDFs to your local `documents/` directory.
```bash
python main.py fetch --query "ti:stable diffusion AND abs:video generation" --max-results 5
```

### 2. Ingest into Vector DB
Process the downloaded PDFs, create embeddings, and store them in Qdrant.
```bash
# Using Ollama (default)
python main.py ingest --provider ollama

# Using Gemini
python main.py ingest --provider gemini --batch-size 10
```

### 3. Query the RAG System
Ask questions based on the ingested research papers.
```bash
python main.py rag --query "What are the latest techniques for video generation in stable diffusion?" --provider gemini
```

## 📊 Logging

The system uses structured logging to provide insights into its operation:
- **`INFO`**: Progress updates, search results, and timing metrics.
- **`WARNING`**: Batch upload retries.
- **`ERROR`**: Fatal ingestion failures.

Logging level can be adjusted in `config/settings.py` or via environment variables (`LOG_LEVEL`).

## ⚙️ Configuration

Configurations are managed via Pydantic models. You can override any setting using environment variables with the `__` delimiter for nested models (e.g., `GEMINI__API_KEY`).
