from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
# qdrant_api_key = os.getenv("QDRANT_API_KEY")
# api_key = os.getenv("GEMINI_API_KEY")

model = ChatOllama(model="gemma3:1b", temperature=0.0)


def main():
    """
    Main function to set up the RAG chain and answer questions.
    """
    try:
    
        # Initialize the Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=None
        )
        
        # Initialize the embeddings model
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        embedding = OllamaEmbeddings(model="embeddinggemma")
        collection_name = "1984_by_george_orwell"
        
        # Create a Qdrant instance for an existing collection
        qdrant = QdrantVectorStore(
            client=client, 
            collection_name=collection_name, 
            embedding = embedding,
            vector_name="1984-dense-vectors"
        )


        # Initialize the retriever from your Qdrant vector store
        retriever = qdrant.as_retriever()

        # Define the prompt template
        template = """
        Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Build the RAG chain using LCEL
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )


        # Create the query
        query = "Who are the main characters?"

        # Retrieve relevant documents to show the sources
        retrieved_docs = retriever.invoke(query)


        # Generate the final answer using the full RAG chain
        answer = rag_chain.invoke(query)
        print("Answer:", answer)
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()
