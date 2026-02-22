from .ollama_provider import OllamaProvider
from .gemini_provider import GeminiProvider
from .base_provider import BaseProvider
from config.settings import Config, provider_type

class ProviderFactory:
    @staticmethod
    def get_provider(provider: str, config: Config) -> BaseProvider:
        """
        Returns a provider instance based on the configuration.
        """
        if provider == provider_type.OLLAMA:
            return OllamaProvider(config.ollama)
        elif provider == provider_type.GEMINI:
            return GeminiProvider(config.gemini)
        else:
            raise ValueError(f"Unsupported provider type: {provider}")
