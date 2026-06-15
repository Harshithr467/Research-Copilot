import os
import hashlib
import uuid
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configure Google GenAI Client
# Note: Using the new google-genai SDK as google-generativeai is deprecated.
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class ResearchPaper(TypedDict):
    """Type definition for the research paper input dictionary."""
    title: str
    content: str
    author: str
    year: int
    source_file: str

def get_embedding(text: str) -> List[float]:
    """
    Generate vector embedding for a given text chunk.
    
    Uses gemini-embedding-2 with 768 dimensions to match the index mapping.
    """
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={
            "task_type": "retrieval_document",
            "output_dimensionality": 768
        }
    )
    return result.embeddings[0].values

def ingest_paper(paper: ResearchPaper) -> dict:
    """
    Splits a research paper into chunks, generates embeddings, and indexes them into Elasticsearch.

    This function serves as the primary entry point for ingesting research papers into the 
    'research_papers' index. It handles text chunking, vector embedding generation via 
    Google Gemini, and metadata preservation.

    Args:
        paper (ResearchPaper): A dictionary containing the following keys:
            title (str): The full title of the paper.
            content (str): The raw text content of the paper.
            author (str): The primary author or citation string.
            year (int): Year of publication.
            source_file (str): The filename of the original document (e.g., "paper.pdf").

    Returns:
        dict: A summary of the ingestion process.
            status (str): "success" or "error".
            chunks_indexed (int): Number of chunks successfully pushed to Elasticsearch.
            parent_id (str): The unique identifier generated for this paper.

    Example:
        >>> from ingester import ingest_paper
        >>> paper_data = {
        ...     "title": "Example Paper",
        ...     "content": "Full text here...",
        ...     "author": "John Doe",
        ...     "year": 2024,
        ...     "source_file": "example.pdf"
        ... }
        >>> result = ingest_paper(paper_data)
        >>> print(result)
        {'status': 'success', 'chunks_indexed': 5, 'parent_id': '...'}
    """
    try:
        # 1. Connect to local Elasticsearch
        es = Elasticsearch(
            ["http://localhost:9200"],
            verify_certs=False,
            request_timeout=30
        )
        index_name = "research_papers"

        # 2. Generate unique parent_id (Deterministic hash of title)
        title_hash = hashlib.md5(paper['title'].encode()).hexdigest()[:10]
        parent_id = f"paper_{title_hash}"

        # 3. Initialize text splitter (500 chars, 100 overlap)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

        # 4. Split content into chunks
        chunks = text_splitter.split_text(paper['content'])
        
        chunks_indexed = 0
        for i, chunk_text in enumerate(chunks):
            # 5. Generate embedding
            embedding = get_embedding(chunk_text)
            
            # 6. Prepare and index document
            # Use deterministic ID to prevent duplicates if script is re-run
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

        return {
            "status": "success",
            "chunks_indexed": chunks_indexed,
            "parent_id": parent_id
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Example usage for testing
    test_paper: ResearchPaper = {
        "title": "Modular Ingestion Pipeline Test",
        "content": """
        The goal of this pipeline is to provide a clean interface for data engineers.
        By modularizing the ingestion process, we ensure that changes to the embedding model
        or chunking strategy are isolated from the extraction logic.
        
        This multi-paragraph content will be split into multiple chunks by the RecursiveCharacterTextSplitter.
        Each chunk will then be processed individually to generate a high-dimensional vector
        representing its semantic meaning in the vector space.
        """,
        "author": "System Test",
        "year": 2026,
        "source_file": "test_pipeline.pdf"
    }
    
    print("Starting test ingestion...")
    summary = ingest_paper(test_paper)
    print(f"Ingestion Summary: {summary}")
