"""
Embedding service using sentence-transformers.
Lazy-loads the model on first use to save memory at startup.
"""

import numpy as np
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global model reference (lazy loaded)
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import get_settings
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts into vectors.

    Args:
        texts: List of text strings to embed

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        batch_size=32,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query text.

    Args:
        query: The search query string

    Returns:
        numpy array of shape (1, embedding_dim)
    """
    return embed_texts([query])
