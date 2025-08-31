from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re
import time

# Load environment variables from .env file
load_dotenv()

# Now you can access it
api_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google_genai",
    api_key=api_key,
    temperature=0.0
)


def preprocess_text(text):
    """
    Cleans the extracted PDF text by:
    - Merging hyphenated words.
    - Removing excessive newlines and whitespace.
    """
    # Merge hyphenated words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    return text.strip()


# --- Local PDF Loader Implementation ---
# 1. Make sure you have a file named 'local_document.pdf' in the same folder.
# NOTE: You will need to install the 'pypdf' library for this to work.
# Run this command in your terminal: pip install pypdf
file_path = r"C:\Users\Ramzan.Agriya\Documents\Data_Books\Introduction.to.Algorithms.4th.Leiserson.Stein.Rivest.Cormen.MIT.Press.9780262046305.EBooksWorld.ir.pdf"

# 2. Initialize the loader with the local file path.
try:
    loader = PyPDFLoader(file_path)

    # --- 2. Load and Split by Page ---
    print(f"Loading data from {file_path}...")
    pages = loader.load_and_split()
    print("Data loaded successfully!")

    if pages:
        print(f"\nPDF has {len(pages)} pages.")

        # --- 3. Preprocessing/Cleaning Step ---
        print("\n--- Preprocessing text ---")
        
        # Store original text for comparison
        original_text_snippet = pages[0].page_content[:500]

        for i, page in enumerate(pages):
            pages[i].page_content = preprocess_text(page.page_content)
        
        print("Preprocessing complete!")

        # --- 4. Chunking Documents ---
        print("\n--- Chunking documents ---")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        
        chunks = text_splitter.split_documents(pages)
        
        print(f"Split the {len(pages)} pages into {len(chunks)} chunks.")

        # --- 5. Create Embeddings & Store in Qdrant Cluster (with Retry Logic) ---
        print("\n--- Creating embeddings and storing in Qdrant ---")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        collection_name = "my_algorithms_book"
        batch_size = 16

        # Use Qdrant.from_documents to create the collection and upload the first batch.
        # This is a robust way to ensure the collection exists before adding more docs.
        print("Initializing Qdrant collection and uploading the first batch...")
        qdrant = Qdrant.from_documents(
            documents=chunks[:batch_size],
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
        )
        print("Initial batch uploaded successfully.")
        
        print(f"Uploading remaining chunks to Qdrant in batches of {batch_size}...")

        # Loop through the rest of the chunks and add them in batches
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # --- Retry Logic ---
            max_retries = 5
            retries = 0
            while retries < max_retries:
                try:
                    qdrant.add_documents(batch)
                    print(f"Uploaded batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                    break # Exit the retry loop on success
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    print(f"Batch upload failed with error: {e}. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
            
            if retries == max_retries:
                print(f"Failed to upload batch {i//batch_size + 1} after {max_retries} attempts. Aborting.")
                break # Exit the main loop if a batch fails repeatedly

        print("\nEmbeddings created and stored in your Qdrant cluster successfully!")
        print(f"You can now query the '{collection_name}' collection.")

    else:
        print("No data was loaded from the file.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the file path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")




# --- 6. Create a RAG Chain for Question Answering ---
print("\n--- Setting up the RAG chain ---")

# 1. Initialize the retriever from your Qdrant vector store
retriever = qdrant.as_retriever()

# 2. Define the prompt template
template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Build the RAG chain using LCEL
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print("RAG chain is ready. You can now ask questions.")

# --- 7. Ask a Question ---
# Your original query
query = "What is a hash table?"
print(f"\nQuery: '{query}'")

# 1. First, retrieve the relevant documents from Qdrant
print("\n--- Retrieving relevant documents ---")
retrieved_docs = retriever.invoke(query)

# 2. Inspect the retrieved documents
print(f"Found {len(retrieved_docs)} relevant documents.")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Source: Page {doc.metadata.get('page', 'N/A')}")
    print(f"Content: {doc.page_content[:500]}...") # Print the first 500 characters

# 3. Now, get the answer from the LLM using the retrieved context
# (This part is optional if you only want to see the sources)
print("\n--- Generating answer based on retrieved documents ---")
answer = rag_chain.invoke(query) # This re-runs the retrieval, but is simple to demonstrate

print("\n--- Final Answer ---")
print(answer)