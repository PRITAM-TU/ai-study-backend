"""
RAG service: orchestrates retrieval-augmented generation.
Combines vector search results with LLM to answer questions.
"""

import logging
from app.rag.vectorstore import VectorStoreManager
from app.ai_features.llm_client import ask_llm

logger = logging.getLogger(__name__)


async def rag_query(
    user_id: int,
    question: str,
    doc_id: int | None = None,
    top_k: int = 5,
) -> dict:
    """
    Answer a question using RAG (Retrieval-Augmented Generation).

    1. Search vector store for relevant chunks
    2. Build context from top results
    3. Send to LLM with system prompt
    4. Return answer with sources

    Args:
        user_id: The authenticated user's ID
        question: The user's question
        doc_id: Optional document ID to scope the search
        top_k: Number of chunks to retrieve

    Returns:
        Dict with 'answer', 'sources', and 'context_used'
    """
    vs = VectorStoreManager()

    # Step 1: Retrieve relevant chunks
    results = vs.search(user_id, question, top_k=top_k, doc_id=doc_id)

    if not results:
        return {
            "answer": "I don't have any documents to reference. Please upload a document first, then ask your question.",
            "sources": [],
            "context_used": False,
        }

    # Step 2: Build context
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[Source {i} | Doc {r['doc_id']} | Relevance: {r['score']:.2f}]\n{r['text']}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Create prompt
    system_prompt = """You are an expert AI study companion. Answer the student's question using ONLY the provided context from their study materials. 

Rules:
- Be accurate and thorough
- If the context doesn't contain enough information, say so clearly
- Use clear explanations suitable for studying
- When helpful, use bullet points, numbered lists, or structured formatting
- Reference which source material you're drawing from
- If asked to explain a concept, make it easy to understand"""

    user_prompt = f"""Context from study materials:

{context}

---

Student's Question: {question}

Please provide a comprehensive answer based on the context above."""

    # Step 4: Get LLM response
    answer = await ask_llm(system_prompt=system_prompt, user_prompt=user_prompt)

    return {
        "answer": answer,
        "sources": [
            {"doc_id": r["doc_id"], "chunk_index": r["chunk_index"], "score": r["score"]}
            for r in results
        ],
        "context_used": True,
    }
