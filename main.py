import argparse
import logging
from config.settings import config, provider_type
from config.logging_utils import setup_logging
from services import ArxivService, IngestService, RagService, StorageService

logger = logging.getLogger(__name__)

def main():
    setup_logging(config.log_level)
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=[p.value for p in provider_type], default="ollama", help="LLM provider to use")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # Fetch Research Papers
    fetch_parser = subparsers.add_parser("fetch", help="Fetch research papers from arXiv")
    fetch_parser.add_argument("--query", "-q", type=str, required=True, help="Search query for fetching papers")
    fetch_parser.add_argument("--max-results", "-m", type=int, default=config.arxiv.max_results, help="Maximum number of papers to fetch")
    fetch_parser.add_argument("--output", "-o", type=str, default=config.file_dir, help="Output file to save fetched papers")

    # Ingest Research Papers
    ingest_parser = subparsers.add_parser("ingest", help="Ingest research papers into vector database")
    ingest_parser.add_argument("--chunk-size", type=int, default=config.vectordb.chunk_size, help="Chunk size for splitting documents")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=config.vectordb.chunk_overlap, help="Chunk overlap for splitting documents")
    ingest_parser.add_argument("--batch-size", type=int, default=config.vectordb.batch_size, help="Batch size for vectorization and ingestion")

    # RAG Query
    rag_parser = subparsers.add_parser("rag", help="Run RAG query against vector database")
    rag_parser.add_argument("--query", "-q", type=str, required=True, help="Query for RAG retrieval")

    args = parser.parse_args()
    provider = provider_type(args.provider)

    # Map args to config fields for automatic update
    arg_mapping = {
        "max_results": (config.arxiv, "max_results"),
        "output": (config, "file_dir"),
        "chunk_size": (config.vectordb, "chunk_size"),
        "chunk_overlap": (config.vectordb, "chunk_overlap"),
        "batch_size": (config.vectordb, "batch_size"),
    }

    for arg_key, (obj, attr) in arg_mapping.items():
        if hasattr(args, arg_key):
            setattr(obj, attr, getattr(args, arg_key))

    # Route to the correct service
    if args.command == "fetch":
        logger.info(f"Fetching {args.max_results} papers for query: '{args.query}'...")
        arxiv_svc = ArxivService(config)
        storage_svc = StorageService(config)
        papers = arxiv_svc.run_service(args.query)
        storage_svc.upload_file(papers)

    elif args.command == "ingest":
        logger.info("Starting ingestion pipeline...")
        ingest_svc = IngestService(config=config, provider=provider)
        ingest_svc.run_pipeline()

    elif args.command == "rag":
        logger.info(f"Querying: {args.query}")
        rag_svc = RagService(config=config, provider=provider)
        answer = rag_svc.query(args.query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()