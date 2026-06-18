import json
from collections.abc import Iterable

from .models import RetrievedChunk


def build_grounded_prompt(question: str, chunks: Iterable[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    for chunk in chunks:
        page = chunk.page_number if chunk.page_number is not None else "unknown"
        source = chunk.source_file or "unknown"
        context_blocks.append(
            "\n".join(
                [
                    f"[chunk_id: {chunk.chunk_id}]",
                    f"[source_file: {source}]",
                    f"[page_number: {page}]",
                    chunk.text.strip(),
                ]
            )
        )

    schema = {
        "answer": "Grounded answer using only the retrieved chunks, or a sentence saying there is not enough evidence.",
        "citations": [
            {
                "source_file": "source file from the cited chunk",
                "page_number": "page number from the cited chunk, or null if absent",
                "chunk_id": "exact chunk_id from the cited chunk",
                "quote": "exact supporting quote copied from the cited chunk",
            }
        ],
    }

    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are Research Copilot, a grounded assistant for a student's uploaded document.

Answer the student's question using only the retrieved document chunks below.
Do not use outside knowledge.
Every factual claim in the answer must be supported by at least one citation.
Use exact quotes from the chunks as citation proof.
If the chunks do not contain enough evidence, answer: "I cannot find enough evidence in the uploaded document to answer that."
Return only valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Student question:
{question}

Retrieved chunks:
{context}
"""
