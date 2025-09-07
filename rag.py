from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import qdrant_client


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


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
     
        # Initialize the Qdrant client
        client = qdrant_client.QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key
        )
        
        # Initialize the embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        collection_name = ""
        
        # Create a Qdrant instance for an existing collection
        qdrant = Qdrant(
            client=client, 
            collection_name=collection_name, 
            embeddings=embeddings
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
        query = ""

        # Retrieve relevant documents to show the sources
        retrieved_docs = retriever.invoke(query)


        # Generate the final answer using the full RAG chain
        answer = rag_chain.invoke(query)


if __name__ == "__main__":
    main()
