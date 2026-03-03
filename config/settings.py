from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from enum import Enum
import os


# Load environment variables from .env file
load_dotenv()

# LLM Provider settings
class provider_type(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"

class ProviderConfig(BaseModel):
    chat_model: str
    embedding_model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=2048, ge=1, le=4096)


# Vector Database settings
class VectorDBConfig(BaseModel):
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection_name: str = "default_collection"
    dense_vector_name: str = "dense_vectors"
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)
    batch_size: int = Field(default=32, ge=1)


# Arxiv settings
class ArxivConfig(BaseModel):
    query: str = ""
    max_results: int = Field(default=200, ge=1)
    sort_by: str = "Relevance"
    sort_order: str = "Descending"


# File Upload settings
class FileUploadConfig(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str


# Log Level
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Main Config
class Config(BaseSettings):
    # provider: provider_type = provider_type.OLLAMA
    
    # Provider-specific configurations
    ollama: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        chat_model="gemma3:4b",
        embedding_model="embeddinggemma"
    ))
    gemini: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        chat_model="gemini-2.5-flash-lite",
        embedding_model="gemini-embedding-001"
    ))

    log_level: LogLevel = LogLevel.INFO

    # Vector Database Configurations
    vectordb: VectorDBConfig = Field(default_factory=lambda: VectorDBConfig(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key = os.getenv("QDRANT_API_KEY"),
        collection_name = "arxiv_collection",
        dense_vector_name = "dense_vectors",
        chunk_size = 1000,
        chunk_overlap = 200,
        batch_size = 32
    ))

    #Arxiv Papers Downloading Configurations
    arxiv: ArxivConfig = Field(default_factory=lambda: ArxivConfig(
        query="ti:stable diffusion AND abs:video generation",
    ))

    minio: FileUploadConfig = Field(default_factory=lambda: FileUploadConfig(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        bucket_name=os.getenv("MINIO_BUCKET_NAME")
    ))


    # file paths
    file_dir: str

    # This reserved dictionary tells Pydantic to look for a .env file
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        env_nested_delimiter="__", # Allows parsing nested models via env vars
        extra="ignore"
    )


config = Config()