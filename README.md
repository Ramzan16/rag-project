# RAG Research Assistant

A CLI-based Retrieval-Augmented Generation (RAG) system designed to fetch, ingest, and query research papers from arXiv using modern LLM providers, vector databases, and S3-compatible storage.

## 🚀 Features

- **arXiv Integration**: Search and stream research papers directly from the arXiv API.
- **S3-Compatible Storage**: Uses **MinIO** to store and manage PDF documents, ensuring a cloud-native and scalable architecture.
- **Batch Data Ingestion**: Robust pipeline for loading PDFs from MinIO, cleaning text, chunking, and batch-uploading to a vector database.
- **Multi-Provider Support**: Switch seamlessly between local (**Ollama**) and cloud-based (**Google Gemini**) LLM providers.
- **Vector Database**: Integration with **Qdrant** for high-performance similarity searches.
- **Structured Logging**: Comprehensive logging for tracking ingestion progress, retrieval times, and generation metrics.
- **Dockerized Infrastructure**: Ready-to-use Docker Compose setup for all core services (Qdrant, Ollama, MinIO).

## 🏗️ Architecture

- **`config/`**: Centralized configuration management using `pydantic-settings`. Supports nested environment variables.
- **`llm_provider/`**: Abstraction layer for LLM interactions (Chat and Embeddings) using a Factory pattern.
- **`services/`**:
  - `ArxivService`: Handles paper discovery and streaming from arXiv.
  - `StorageService`: Manages file persistence in MinIO.
  - `IngestService`: Orchestrates the document-to-vector pipeline (Load from MinIO -> Chunk -> Embed -> Qdrant).
  - `RagService`: Manages retrieval-augmented generation logic.
- **`main.py`**: Unified CLI entry point for all operations.

## 📁 Project Structure

```text
.
├── config/                 # Configuration and logging utilities
├── llm_provider/           # LLM provider implementations (Gemini, Ollama)
├── services/               # Core business logic (Arxiv, Ingest, RAG, Storage)
├── tests/                  # Unit and integration tests
├── main.py                 # CLI Entry point
├── schemas.py              # Pydantic models for data exchange
├── docker-compose.yml      # Infrastructure orchestration
├── Dockerfile              # Application containerization
└── pyproject.toml          # Dependency management (uv)
```

## 🛠️ Setup

### Prerequisites

- **Python**: 3.12+
- **Docker & Docker Compose**: For running infrastructure services.
- **Package Manager**: [uv](https://github.com/astral-sh/uv) is recommended.

### 1. Infrastructure Setup

Start the core services (Qdrant, Ollama, MinIO) using Docker Compose:

```bash
docker-compose up -d qdrant ollama minio
```

### 2. Application Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ramzan16/rag-project.git
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
    Update the values in `.env` with your API keys (e.g., `GEMINI__API_KEY`) and service URLs.

## 📖 Usage

The CLI supports three primary commands: `fetch`, `ingest`, and `rag`.

### 1. Fetch Research Papers
Search arXiv and stream PDFs directly into your **MinIO** bucket.
```bash
# Fetch papers related to Stable Diffusion
python main.py fetch --query "ti:stable diffusion" --max-results 5
```

### 2. Ingest into Vector DB
Pull PDFs from MinIO, process them, and store embeddings in **Qdrant**.
```bash
# Ingest using Ollama (default)
python main.py ingest --provider ollama

# Ingest using Gemini with custom batch size
python main.py ingest --provider gemini --batch-size 10
```

### 3. Query the RAG System
Ask questions based on the ingested research papers.
```bash
python main.py rag --query "What are the latest techniques for video generation?" --provider gemini
```

## 📊 Logging

The system uses structured logging:
- **`INFO`**: Progress updates and timing metrics.
- **`WARNING`**: Retry attempts for batch uploads.
- **`ERROR`**: Fatal failures in the pipeline.

Adjust levels via `LOG_LEVEL` in `.env`.

## ⚙️ Configuration

Configurations are managed via Pydantic. Override any setting using environment variables with the `__` delimiter for nested models:
- `GEMINI__API_KEY`: API key for Google Gemini.
- `MINIO__ENDPOINT`: URL for the MinIO service.
- `VECTORDB__COLLECTION_NAME`: Target Qdrant collection.
