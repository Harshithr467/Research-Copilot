import os
import hashlib
import uuid
import threading
import time
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# Global singleton and a lock to ensure thread safety
_embedding_model = None
_model_ready_event = threading.Event()

def _load_model_background():
    """Heavy initialization performed in a background thread."""
    global _embedding_model
    try:
        # Import inside the thread to keep the main import fast
        from langchain_huggingface import HuggingFaceEmbeddings
        print("\n[Background] Starting to load embedding model (all-MiniLM-L6-v2)...")
        
        model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        _embedding_model = model
        _model_ready_event.set() # Signal that the model is ready
        print("[Background] Embedding model loaded successfully.")
    except Exception as e:
        print(f"[Background] Failed to load embedding model: {e}")

# Start the background loading immediately upon module import
# daemon=True ensures this thread doesn't block the server from shutting down
threading.Thread(target=_load_model_background, daemon=True).start()

class ResearchPaper(TypedDict):
    """Type definition for the research paper input dictionary."""
    title: str
    content: str
    author: str
    year: int
    source_file: str

def get_embedding_model():
    """
    Returns the global embedding model. 
    If it's still loading in the background, this will wait until it's ready.
    """
    if not _model_ready_event.is_set():
        print("Waiting for background model to finish loading... (this only happens on the first upload if you're very fast)")
        _model_ready_event.wait()
    return _embedding_model

def ingest_paper(paper: ResearchPaper) -> dict:
    """
    Splits a research paper into chunks, generates embeddings locally, and indexes them into Elasticsearch.
    """
    try:
        # Move heavy import inside the function
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 1. Connect to local Elasticsearch
        es = Elasticsearch(
            ["http://localhost:9200"],
            verify_certs=False,
            request_timeout=30
        )
        index_name = "research_papers"

        # 2. Get the embedding model (waits if background thread isn't done yet)
        embeddings = get_embedding_model()

        # 3. Generate unique parent_id (Deterministic hash of title)
        title_hash = hashlib.md5(paper['title'].encode()).hexdigest()[:10]
        parent_id = f"paper_{title_hash}"

        # 4. Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

        # 5. Split content into chunks
        chunks = text_splitter.split_text(paper['content'])
        total_chunks = len(chunks)
        
        # 6. Generate embeddings for all chunks locally
        print(f"[{paper['source_file']}] Generating local embeddings for {total_chunks} chunks...")
        all_embeddings = embeddings.embed_documents(chunks)
        
        # 7. Index each chunk into Elasticsearch
        chunks_indexed = 0
        for i, (chunk_text, embedding) in enumerate(zip(chunks, all_embeddings)):
            doc_id = f"{parent_id}_chunk_{i}"
            
            doc = {
                "title": paper['title'],
                "content": chunk_text,
                "embedding": embedding,
                "metadata": {
                    "author": paper['author'],
                    "year": paper['year'],
                    "source_file": paper['source_file'],
                    "parent_id": parent_id,
                    "chunk_id": i
                }
            }
            
            es.index(index=index_name, id=doc_id, document=doc)
            chunks_indexed += 1
            if chunks_indexed % 50 == 0:
                print(f"[{paper['source_file']}] Indexed {chunks_indexed}/{total_chunks} chunks...")

        print(f"[{paper['source_file']}] Ingestion complete. Total chunks: {chunks_indexed}")

        return {
            "status": "success",
            "chunks_indexed": chunks_indexed,
            "parent_id": parent_id
        }

    except Exception as e:
        print(f"Ingestion failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Example usage for testing
    test_paper: ResearchPaper = {
        "title": "Local Ingestion Pipeline Test",
        "content": "Sample content for testing ingestion pipeline locally.",
        "author": "System Test",
        "year": 2026,
        "source_file": "test_local_pipeline.pdf"
    }
    summary = ingest_paper(test_paper)
    print(f"Ingestion Summary: {summary}")
