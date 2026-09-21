from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from config.settings import ProviderConfig


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """
        Returns an instance of a chat model based on the provider's configuration.
        """
        raise NotImplementedError("get_chat_model method must be implemented by the provider.")

    @abstractmethod
    def get_embedding_model(self) -> Embeddings:
        """
        Returns an instance of an embedding model based on the provider's configuration.
        """
        raise NotImplementedError("get_embedding_model method must be implemented by the provider.")