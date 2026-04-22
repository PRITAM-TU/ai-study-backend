"""
Text chunking utilities for the RAG pipeline.
Splits document text into overlapping chunks for embedding.
"""

import re
from app.config import get_settings

settings = get_settings()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.

    Strategy:
    1. Split on paragraphs (double newlines)
    2. If a paragraph is too long, split on sentences
    3. If a sentence is too long, split on words
    4. Merge small chunks with overlap

    Args:
        text: The full document text
        chunk_size: Max characters per chunk (default from settings)
        chunk_overlap: Number of overlapping characters (default from settings)

    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # If text fits in one chunk, return as-is
    if len(text) <= chunk_size:
        return [text]

    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If paragraph itself is too long, split it further
        if len(paragraph) > chunk_size:
            # Split on sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # If sentence itself is too long, split on words
                    if len(sentence) > chunk_size:
                        words = sentence.split()
                        current_chunk = ""
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= chunk_size:
                                current_chunk = f"{current_chunk} {word}".strip() if current_chunk else word
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = word
                    else:
                        current_chunk = sentence
        else:
            # Try to add paragraph to current chunk
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Apply overlap: prepend the tail of the previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped_chunks.append(f"{prev_tail}... {chunks[i]}")
        return overlapped_chunks

    return chunks
