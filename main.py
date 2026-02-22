from llm_provider.factory import ProviderFactory
from config.settings import Config, config

model = ProviderFactory.get_provider('ollama', config)
print(model.get_chat_model())
print(model.get_embedding_model())