from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llm_provider import ProviderFactory
from config.settings import Config, provider_type
from config.settings import config


class RagService:
    """
    Service class to handle the RAG pipeline.
    """
    def __init__(self, config: Config = config, provider: provider_type = provider_type.OLLAMA):
        self.config = config
        self.provider = ProviderFactory.get_provider(provider, config)

    def setup_rag_chain(self, retriever):
        """
        Sets up the RAG chain using LCEL.
        """
        template = """
        Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.provider.get_chat_model()
            | StrOutputParser()
        )
        return rag_chain

    def init_qdrant_instance(self):
        """
        Initializes the Qdrant client using the configurations.
        """
        client = QdrantClient(
            url=self.config.vectordb.qdrant_url,
            api_key=self.config.vectordb.qdrant_api_key
        )
        qdrant = QdrantVectorStore(
            client=client, 
            collection_name=self.config.vectordb.collection_name, 
            embedding=self.provider.get_embedding_model(),
            vector_name=self.config.vectordb.dense_vector_name
        )
        return qdrant

    def query(self, query):
        """
        Executes a query against the RAG chain and returns the answer.
        """
        qdrant = self.init_qdrant_instance()
        retriever = qdrant.as_retriever()
        rag_chain = self.setup_rag_chain(retriever)
        answer = rag_chain.invoke(query)
        return answer