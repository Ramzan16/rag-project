from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_provider import ProviderFactory
from config.settings import Config, provider_type
from config.settings import config
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
import re
import time
import hashlib


class IngestService:
    """
    Service class to handle the data ingestion pipeline.
    """
    def __init__(self, config: Config = config, provider: provider_type = provider_type.OLLAMA):
        self.config = config
        self.provider = ProviderFactory.get_provider(provider, config)

    @staticmethod
    def preprocess_text(text: str):
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


    def load_files(self):
        """
        Loads the pdfs from the directory specified in config using PyPDFLoader.
        """
        project_dir = Path(__file__).resolve().parent.parent
        file_dir = project_dir / self.config.file_dir
        loader = PyPDFDirectoryLoader(
            path=str(file_dir),
            glob='**/[!.]*.pdf',
            recursive = True,
            )
        documents = loader.load()

        #Updating metadata
        for doc in documents:
            doc.page_content = self.preprocess_text(doc.page_content)
            doc.metadata = {
                "doc_id": hashlib.sha1(doc.metadata.get("source", "").encode("utf-8")).hexdigest(),
                "creationdate": doc.metadata.get("creationdate", ""),
                "author": doc.metadata.get("author", ""),
                "moddate": doc.metadata.get("moddate", ""),
                "title": Path(doc.metadata.get("source", "")).name,
                "source": doc.metadata.get("source", ""),
                "total_pages": doc.metadata.get("total_pages", 0),
                "page": doc.metadata.get("page", 0),
                "page_label": doc.metadata.get("page_label", "")
            }
        return documents

    def chunk_documents(self, documents):
        """
        Chunks the documents using RecursiveCharacterTextSplitter.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def embedd_and_store(self, chunks):
        """
        Creates embeddings for the chunks and stores them in Qdrant.
        """
        if self.config.vectordb.qdrant_api_key:
            client = QdrantClient(
                url=self.config.vectordb.qdrant_url,
                api_key=self.config.vectordb.qdrant_api_key
            )
        # For a local Qdrant instance without API key
        else:
            client = QdrantClient(url=self.config.vectordb.qdrant_url)

        qdrant = QdrantVectorStore(
            client=client,
            collection_name=self.config.vectordb.collection_name,
            embedding=self.provider.get_embedding_model(),
            vector_name=self.config.vectordb.dense_vector_name
        )

        qdrant.add_documents(chunks[:self.config.batch_size])

        # Loop through the rest of the chunks and add them in batches with retry logic
        for i in range(self.config.batch_size, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            max_retries = 5
            retries = 0
            while retries < max_retries:
                try:
                    qdrant.add_documents(batch)
                    print(f"Uploaded batch {i//self.config.vectordb.batch_size + 1}/{(len(chunks) + self.config.vectordb.batch_size - 1)//self.config.vectordb.batch_size}")
                    break
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries
                    print(f"Batch upload failed: {e}. Retrying in {wait_time}s... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)

            if retries == max_retries:
                print(f"Failed to upload batch {i//self.config.vectordb.batch_size + 1} after {max_retries} attempts. Aborting.")
                break

    def run_pipeline(self):
        """
        Runs the entire ingestion pipeline: load, preprocess, chunk, embed, and store.
        """
        documents = self.load_files()
        chunks = self.chunk_documents(documents)
        self.embedd_and_store(chunks)