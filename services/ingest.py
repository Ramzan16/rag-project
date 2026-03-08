from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_provider import ProviderFactory
from config.settings import Config, provider_type, config
from services.storage_service import StorageService
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
import re
import time
import hashlib
import logging
import os
import tempfile


logger = logging.getLogger(__name__)


class IngestService:
    """
    Service class to handle the data ingestion pipeline.
    """
    def __init__(self, config: Config = config, provider: provider_type = provider_type.OLLAMA):
        self.config = config
        self.provider = ProviderFactory.get_provider(provider, config)

        # Initialize StorageService to load files
        self.storage_service = StorageService(config)


    @staticmethod
    def _preprocess_text(text: str):
        """
        Cleans the extracted PDF text by:
        - Merging hyphenated words.
        - Removing excessive newlines and whitespace.
        """
        # Merge hyphenated words
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Replace multiple newlines with a single space
        text = re.sub(r'\n+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _get_qdrant_client(self) -> QdrantClient:
        """Helper to initialize the Qdrant client."""
        if self.config.vectordb.qdrant_api_key:
            return QdrantClient(
                url=self.config.vectordb.qdrant_url,
                api_key=self.config.vectordb.qdrant_api_key
            )
        return QdrantClient(url=self.config.vectordb.qdrant_url)


    def _is_already_ingested(self, client: QdrantClient, filename: str) -> bool:
        """Checks if a file has already been ingested into Qdrant."""
        try:
            result = client.count(
                collection_name=self.config.vectordb.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
            return result.count > 0
        except Exception as e:
            # Collection likely does not exist yet
            logger.debug(f"Could not check ingestion status for {filename}. Assuming not ingested. Reason: {e}")
            return False

    def _upload_batch_with_retry(self, qdrant: QdrantVectorStore, batch: list, current_batch_num: int, total_batches: int, max_retries: int = 5) -> bool:
        """
        Attempts to upload a batch of documents to Qdrant, using exponential backoff on failure.
        Returns True if successful, False if all retries are exhausted.
        """
        retries = 0
        while retries < max_retries:
            try:
                qdrant.add_documents(batch)
                logger.info(f"Uploaded batch {current_batch_num}/{total_batches}")
                return True
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries
                logger.warning(f"Batch {current_batch_num} upload failed: {e}. Retrying in {wait_time}s... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)

        logger.error(f"Failed to upload batch {current_batch_num} after {max_retries} attempts. Aborting ingestion loop.")
        return False


    def load_files(self):
        """
        Loads the pdfs from the directory specified in config using PyPDFLoader.
        """
        # Listing and loading files
        logger.info("Loading documents")
        files_list = self.storage_service.list_files()

        all_documents = []
        q_client = self._get_qdrant_client()

        for i, file_name in enumerate(files_list):
            if self._is_already_ingested(q_client, file_name):
                logger.info(f"Skipping file {i+1}/{len(files_list)}: '{file_name}' (Already ingested)")
                continue

            logger.debug(f"Processing file {i+1}/{len(files_list)}: {file_name}")

            # Load file from storage
            file_responses = list(self.storage_service.load_files([file_name]))
            if not file_responses:
                logger.warning(f"Could not load {file_name} from storage.")
                continue

            file_response = file_responses[0]
            file_ext = Path(file_name).suffix

            # Write the MinIO stream directly to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file_response.read())
                tmp_path = tmp_file.name

            try:
                # Pass the physical file path to the loader
                loader = PyPDFLoader(file_path=tmp_path)
                docs = loader.load()

                # Patch the source metadata to reflect the MinIO object, not the temp file
                for doc in docs:
                    if "source" in doc.metadata:
                        doc.metadata["source"] = file_name

                # Flatten the document structure
                all_documents.extend(docs)

            except Exception as e:
                logger.error(f"Error loading {file_name}: {e}")
                continue

            finally:
                # Delete File After loading
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                file_response.close()
                file_response.release_conn()

        logger.info(f"Successfully loaded {len(all_documents)} new document pages.")

        for doc in all_documents:
            doc.page_content = self._preprocess_text(doc.page_content)
            m = doc.metadata
            source_name = m.get("source", "unknown")

            doc.metadata = {
                "doc_id": hashlib.sha1(source_name.encode("utf-8")).hexdigest(),
                "creation_date": m.get("creationdate", ""),
                "author": m.get("author", ""),
                "mod_date": m.get("moddate", ""),
                "title": m.get("title") or Path(source_name).name,
                "source": source_name,
                "total_pages": m.get("total_pages", 0),
                "page": m.get("page", 0),
                "page_label": m.get("page_label", "")
            }

        return all_documents

    def chunk_documents(self, documents):
        if not documents:
            logger.info("No new documents to chunk.")
            return []

        logger.info(f"Chunking {len(documents)} document pages...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.vectordb.chunk_size, 
            chunk_overlap=self.config.vectordb.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def embedd_and_store(self, chunks):
        if not chunks:
            logger.info("No new chunks to embed and store.")
            return

        logger.info(f"Embedding and storing {len(chunks)} chunks in collection: '{self.config.vectordb.collection_name}'")

        client = self._get_qdrant_client()

        try:
            client.get_collection(collection_name=self.config.vectordb.collection_name)
            logger.info(f"Collection '{self.config.vectordb.collection_name}' already exists.")
        except Exception:
            logger.info(f"Collection '{self.config.vectordb.collection_name}' not found. Creating new collection.")
            client.recreate_collection(
                collection_name=self.config.vectordb.collection_name,
                vectors_config={
                    self.config.vectordb.dense_vector_name: models.VectorParams(
                        size=len(self.provider.get_embedding_model().embed_query(".")),
                        distance=models.Distance.COSINE
                    )
                }
            )
            logger.info("Collection created successfully.")

        qdrant = QdrantVectorStore(
            client=client,
            collection_name=self.config.vectordb.collection_name,
            embedding=self.provider.get_embedding_model(),
            vector_name=self.config.vectordb.dense_vector_name
        )

        total_batches = (len(chunks) + self.config.vectordb.batch_size - 1) // self.config.vectordb.batch_size

        for i in range(0, len(chunks), self.config.vectordb.batch_size):
            batch = chunks[i:i + self.config.vectordb.batch_size]
            current_batch_num = (i // self.config.vectordb.batch_size) + 1

            if not self._upload_batch_with_retry(qdrant, batch, current_batch_num, total_batches):
                break

    def run_pipeline(self):
        """
        Runs the entire ingestion pipeline: load, preprocess, chunk, embed, and store.
        """
        documents = self.load_files()
        if documents:
            chunks = self.chunk_documents(documents)
            if chunks:
                self.embedd_and_store(chunks)
        else:
            logger.info("Ingestion pipeline finished. No new documents to process.")