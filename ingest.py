from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
import os
import re
import time

# --- 1. Load Environment Variables ---
# Make sure you have a .env file with your API keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

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

def main():
    """
    Main function to handle the data ingestion pipeline.
    """
    # --- 2. Load and Process the PDF Document ---
    # NOTE: You will need to install the 'pypdf' library for this to work.
    # Run this command in your terminal: pip install pypdf
    file_path = r"C:\Users\Ramzan.Agriya\Documents\Data_Books\Introduction.to.Algorithms.4th.Leiserson.Stein.Rivest.Cormen.MIT.Press.9780262046305.EBooksWorld.ir.pdf"

    try:
        loader = PyPDFLoader(file_path)
        print(f"Loading data from {file_path}...")
        pages = loader.load_and_split()
        print("Data loaded successfully!")

        if not pages:
            print("No data was loaded from the file. Exiting.")
            return

        print(f"\nPDF has {len(pages)} pages.")

        # --- 3. Preprocessing/Cleaning Step ---
        print("\n--- Preprocessing text ---")
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

        # --- 5. Create Embeddings & Store in Qdrant Cluster ---
        print("\n--- Creating embeddings and storing in Qdrant ---")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        collection_name = "my_algorithms_book"
        batch_size = 16

        # Use Qdrant.from_documents to create the collection and upload the first batch.
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

        # Loop through the rest of the chunks and add them in batches with retry logic
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            max_retries = 5
            retries = 0
            while retries < max_retries:
                try:
                    qdrant.add_documents(batch)
                    print(f"Uploaded batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                    break
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries
                    print(f"Batch upload failed: {e}. Retrying in {wait_time}s... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
            
            if retries == max_retries:
                print(f"Failed to upload batch {i//batch_size + 1} after {max_retries} attempts. Aborting.")
                break

        print("\nEmbeddings created and stored successfully!")
        print(f"You can now query the '{collection_name}' collection using the other script.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
