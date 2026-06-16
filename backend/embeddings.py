import os
from typing import List, cast

from dotenv import load_dotenv

load_dotenv()


def embed_query(text: str) -> List[float]:
    """Generate a query embedding that matches the 768-dim index mapping."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for vector retrieval.")

    from google import genai

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-2"),
        contents=text,
        config={
            "task_type": "retrieval_query",
            "output_dimensionality": 768,
        },
    )
    embeddings = result.embeddings
    if not embeddings:
        raise RuntimeError("Google GenAI did not return an embedding.")
    values = embeddings[0].values
    if values is None:
        raise RuntimeError("Google GenAI did not return an embedding.")
    return cast(List[float], values)
