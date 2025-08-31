from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant  #type: ignore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import qdrant_client  #type: ignore

# --- 1. Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# --- 2. Initialize the LLM ---
model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google_genai",
    api_key=api_key,
    temperature=0.0
)

def main():
    """
    Main function to set up the RAG chain and answer questions.
    """
    try:
        # --- 3. Connect to Existing Qdrant Collection ---
        print("--- Connecting to Qdrant vector store ---")
        
        # Initialize the Qdrant client
        client = qdrant_client.QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key
        )
        
        # Initialize the embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        collection_name = "my_algorithms_book"
        
        # Create a Qdrant instance for an existing collection
        qdrant = Qdrant(
            client=client, 
            collection_name=collection_name, 
            embeddings=embeddings
        )
        
        print(f"Successfully connected to the '{collection_name}' collection.")

        # --- 4. Create a RAG Chain for Question Answering ---
        print("\n--- Setting up the RAG chain ---")

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

        print("RAG chain is ready. You can now ask questions.")

        # --- 5. Ask a Question ---
        query = "What is a hash table?"
        print(f"\nQuery: '{query}'")

        # Retrieve relevant documents to show the sources
        print("\n--- Retrieving relevant documents ---")
        retrieved_docs = retriever.invoke(query)
        
        print(f"Found {len(retrieved_docs)} relevant documents.")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Source: Page {doc.metadata.get('page', 'N/A')}")
            print(f"Content snippet: {doc.page_content[:500]}...")

        # Generate the final answer using the full RAG chain
        print("\n--- Generating answer based on retrieved documents ---")
        answer = rag_chain.invoke(query)

        print("\n--- Final Answer ---")
        print(answer)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the 'my_algorithms_book' collection exists and the Qdrant server is running.")

if __name__ == "__main__":
    main()
