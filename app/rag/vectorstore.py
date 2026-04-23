"""
FAISS vector store manager.
Handles creating, loading, searching, and persisting vector indexes.
Each user gets their own index to isolate data.
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.rag.embeddings import embed_texts, embed_query

settings = get_settings()
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages FAISS vector indexes per user."""

    def __init__(self):
        self.store_dir = settings.VECTOR_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = settings.EMBEDDING_DIMENSION

    def _index_path(self, user_id: str) -> Path:
        return self.store_dir / f"user_{user_id}.index"

    def _metadata_path(self, user_id: str) -> Path:
        return self.store_dir / f"user_{user_id}_meta.json"

    def _load_index(self, user_id: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
        """Load existing FAISS index and metadata for a user."""
        index_path = self._index_path(user_id)
        meta_path = self._metadata_path(user_id)

        if index_path.exists() and meta_path.exists():
            index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return index, metadata
        else:
            # Create new index (Inner Product for cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(self.dimension)
            return index, []

    def _save_index(self, user_id: str, index: faiss.IndexFlatIP, metadata: list[dict]):
        """Persist FAISS index and metadata to disk."""
        faiss.write_index(index, str(self._index_path(user_id)))
        with open(self._metadata_path(user_id), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

    async def add_document_chunks(
        self,
        doc_id: str,
        user_id: str,
        chunks: list[str],
    ):
        """
        Embed and add document chunks to the user's vector store.

        Args:
            doc_id: Database document ID
            user_id: Owner user ID
            chunks: List of text chunks from the document
        """
        if not chunks:
            return

        # Embed all chunks
        embeddings = embed_texts(chunks)

        # Load existing index
        index, metadata = self._load_index(user_id)

        # Add to FAISS
        index.add(embeddings)

        # Add metadata for each chunk
        for i, chunk in enumerate(chunks):
            metadata.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk,
            })

        # Save
        self._save_index(user_id, index, metadata)
        logger.info(f"Added {len(chunks)} chunks for doc {doc_id} to user {user_id}'s vector store")

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[dict]:
        """
        Search the user's vector store for relevant chunks.

        Args:
            user_id: Owner user ID
            query: Search query text
            top_k: Number of results to return
            doc_id: Optional filter to search only within one document

        Returns:
            List of dicts with keys: doc_id, chunk_index, text, score
        """
        top_k = top_k or settings.RAG_TOP_K
        index, metadata = self._load_index(user_id)

        if index.ntotal == 0:
            return []

        # Embed query
        query_vector = embed_query(query)

        # Search FAISS
        scores, indices = index.search(query_vector, min(top_k * 3, index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = metadata[idx]

            # Filter by doc_id if specified
            if doc_id is not None and entry["doc_id"] != doc_id:
                continue

            results.append({
                "doc_id": entry["doc_id"],
                "chunk_index": entry["chunk_index"],
                "text": entry["text"],
                "score": float(score),
            })

            if len(results) >= top_k:
                break

        return results

    def get_all_chunks(self, user_id: str, doc_id: str) -> list[str]:
        """Get all text chunks for a specific document."""
        _, metadata = self._load_index(user_id)
        chunks = [
            m["text"] for m in metadata
            if m["doc_id"] == doc_id
        ]
        return sorted(chunks, key=lambda x: metadata[[
            m["text"] for m in metadata if m["doc_id"] == doc_id
        ].index(x) if x in [m["text"] for m in metadata if m["doc_id"] == doc_id] else 0])

    def get_document_text(self, user_id: str, doc_id: str) -> str:
        """Get the full reconstructed text for a document from its chunks."""
        _, metadata = self._load_index(user_id)
        doc_chunks = sorted(
            [m for m in metadata if m["doc_id"] == doc_id],
            key=lambda x: x["chunk_index"],
        )
        return "\n\n".join(c["text"] for c in doc_chunks)

    def remove_document(self, doc_id: str, user_id: str):
        """
        Remove all chunks for a document from the vector store.
        Note: FAISS doesn't support deletion, so we rebuild the index.
        """
        index, metadata = self._load_index(user_id)

        # Filter out chunks belonging to this document
        new_metadata = [m for m in metadata if m["doc_id"] != doc_id]

        if len(new_metadata) == len(metadata):
            return  # Nothing to remove

        # Rebuild index with remaining chunks
        new_index = faiss.IndexFlatIP(self.dimension)
        if new_metadata:
            texts = [m["text"] for m in new_metadata]
            embeddings = embed_texts(texts)
            new_index.add(embeddings)

        self._save_index(user_id, new_index, new_metadata)
        logger.info(f"Removed doc {doc_id} from user {user_id}'s vector store")
