import os
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# Increase pypdf decompression limit to handle large PDFs
import pypdf.filters
pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = 0

from config import Config

def create_vector_db():
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR)
        print(f"Created {Config.DATA_DIR} directory. Please add PDF or TXT files.")
        return

    print(f"Loading documents from {Config.DATA_DIR}...")
    
    documents = []
    
    # Load PDFs
    pdf_loader = DirectoryLoader(Config.DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    if pdf_docs:
        print(f"Loaded {len(pdf_docs)} PDF pages.")
        documents.extend(pdf_docs)
        
    # Load TXT files (keep backward compatibility)
    txt_loader = DirectoryLoader(Config.DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    if txt_docs:
        print(f"Loaded {len(txt_docs)} TXT files.")
        documents.extend(txt_docs)

    if not documents:
        print("No documents found to ingest.")
        return

    print(f"Splitting text (Chunk Size: {Config.CHUNK_SIZE}, Overlap: {Config.CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print(f"Generating embeddings using {Config.EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    
    print("Creating vector store...")
    db = FAISS.from_documents(texts, embeddings)
    
    print(f"Saving vector store to {Config.VECTOR_STORE_PATH}...")
    db.save_local(Config.VECTOR_STORE_PATH)
    print("Ingestion complete.")

if __name__ == "__main__":
    create_vector_db()
