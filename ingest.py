import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def run_ingestion():
    pdf_path = "./data/Ebook-Agentic-AI.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find {pdf_path}")
        return

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(" Initializing ChromaDB and ingesting documents...")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="agentic_ai_collection"
    )
    
    print(f"Successfully ingested {len(chunks)} chunks into './chroma_db'!")

if __name__ == "__main__":
    run_ingestion()