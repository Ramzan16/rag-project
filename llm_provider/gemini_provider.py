from .base_provider import BaseProvider
from config.settings import ProviderConfig, config
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


class GeminiProvider(BaseProvider):
    def __init__(self, config: ProviderConfig = config.gemini):
        super().__init__(config)

    def get_chat_model(self) -> BaseChatModel:
        """
        Returns an instance of the ChatGoogleGenerativeAI model.
        """
        return ChatGoogleGenerativeAI(
            model=self.config.chat_model,
            google_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def get_embedding_model(self) -> Embeddings:
        """
        Returns an instance of the GoogleGenerativeAIEmbeddings model.
        """
        return GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model,
            google_api_key=self.config.api_key,
        )
