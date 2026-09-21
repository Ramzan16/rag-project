import argparse
import logging
from src.config.settings import config, provider_type
from src.config.logging_utils import setup_logging
from src.services import ArxivService, IngestService, RagService, StorageService, OpenAlexService

logger = logging.getLogger(__name__)

def main():
    setup_logging(config.log_level)
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=[p.value for p in provider_type], default="ollama", help="LLM provider to use")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")


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

    # Route to the correct service
    if args.command == "fetch":
        config.arxiv.max_results = args.max_results
        logger.info(f"Fetching {args.max_results} papers from arXiv for query: '{args.query}'...")
        arxiv_svc = ArxivService(config)
        storage_svc = StorageService(config)
        papers = arxiv_svc.run_service(args.query)
        for paper in papers:
            storage_svc.upload_file(paper)

    elif args.command == "ingest":
        config.vectordb.chunk_size = args.chunk_size
        config.vectordb.chunk_overlap = args.chunk_overlap
        config.vectordb.batch_size = args.batch_size
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
