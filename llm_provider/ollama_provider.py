from langchain_ollama import ChatOllama, OllamaEmbeddings
from .base_provider import BaseProvider
from config.settings import ProviderConfig, config
import logging

logger = logging.getLogger(__name__)

class OllamaProvider(BaseProvider):
    def __init__(self, config: ProviderConfig = config.ollama):
        super().__init__(config)
    
    def get_chat_model(self) -> ChatOllama:
        """
        Returns an instance of the ChatOllama model based on the provider's configuration.
        """
        logger.info(f"Initializing Ollama Chat Model: {self.config.chat_model}")
        return ChatOllama(
            model=self.config.chat_model, 
            temperature=self.config.temperature, 
            max_tokens=self.config.max_tokens,
            base_url=self.config.base_url
        )

    def get_embedding_model(self) -> OllamaEmbeddings:
        """
        Returns an instance of the OllamaEmbeddings model based on the provider's configuration.
        """
        logger.info(f"Initializing Ollama Embedding Model: {self.config.embedding_model}")
        return OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.base_url
        )

