"""
Document service: handles upload, parsing, and storage logic.
"""

import uuid
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.documents.models import Document
from app.documents.parser import parse_document
from app.rag.chunker import chunk_text
from app.rag.vectorstore import VectorStoreManager

settings = get_settings()
logger = logging.getLogger(__name__)


async def save_uploaded_file(file_content: bytes, original_name: str) -> tuple[Path, str]:
    """
    Save an uploaded file to disk.
    Returns (file_path, file_type).
    """
    # Determine file type
    suffix = Path(original_name).suffix.lower()
    type_map = {".pdf": "pdf", ".pptx": "pptx", ".txt": "txt", ".text": "txt"}
    file_type = type_map.get(suffix)
    if not file_type:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .pptx, .txt")

    # Create upload directory
    upload_dir = settings.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save with unique name
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    file_path = upload_dir / unique_name
    file_path.write_bytes(file_content)

    return file_path, file_type


async def process_document(
    db: AsyncSession,
    user_id: int,
    file_content: bytes,
    original_name: str,
) -> Document:
    """
    Full document processing pipeline:
    1. Save file to disk
    2. Parse text content
    3. Create DB record
    4. Chunk text and add to vector store
    """
    # Step 1: Save file
    file_path, file_type = await save_uploaded_file(file_content, original_name)

    # Step 2: Create document record (status: processing)
    doc = Document(
        user_id=user_id,
        filename=file_path.name,
        original_name=original_name,
        file_type=file_type,
        file_size=len(file_content),
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        # Step 3: Parse text
        text_content = parse_document(file_path, file_type)
        doc.text_content = text_content

        # Step 4: Chunk and embed
        chunks = chunk_text(text_content)
        doc.chunk_count = len(chunks)

        # Add chunks to vector store
        vs_manager = VectorStoreManager()
        await vs_manager.add_document_chunks(
            doc_id=doc.id,
            user_id=user_id,
            chunks=chunks,
        )

        doc.status = "ready"
        logger.info(f"Document processed: {original_name} ({len(chunks)} chunks)")

    except Exception as e:
        doc.status = "failed"
        logger.error(f"Document processing failed: {e}")
        raise

    finally:
        await db.commit()
        await db.refresh(doc)

    return doc


async def get_user_documents(db: AsyncSession, user_id: int) -> list[Document]:
    """Get all documents for a user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def get_document_by_id(db: AsyncSession, doc_id: int, user_id: int) -> Document | None:
    """Get a specific document by ID, scoped to user."""
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def delete_document(db: AsyncSession, doc_id: int, user_id: int) -> bool:
    """Delete a document and its vector store data."""
    doc = await get_document_by_id(db, doc_id, user_id)
    if not doc:
        return False

    # Remove from vector store
    vs_manager = VectorStoreManager()
    vs_manager.remove_document(doc_id, user_id)

    # Remove file from disk
    file_path = settings.UPLOAD_DIR / doc.filename
    if file_path.exists():
        file_path.unlink()

    # Remove from DB
    await db.delete(doc)
    await db.commit()
    return True
