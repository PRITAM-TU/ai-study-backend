"""
Document service: handles upload, parsing, and storage logic.
"""

import uuid
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from app.config import get_settings
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


def _format_document(doc: dict | None) -> dict | None:
    if not doc:
        return None
    doc["id"] = str(doc.pop("_id"))
    return doc


async def process_document(
    db: AsyncIOMotorDatabase,
    user_id: str,
    file_content: bytes,
    original_name: str,
) -> dict:
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
    doc_dict = {
        "user_id": user_id,
        "filename": file_path.name,
        "original_name": original_name,
        "file_type": file_type,
        "file_size": len(file_content),
        "text_content": None,
        "chunk_count": 0,
        "status": "processing",
        "created_at": datetime.now(timezone.utc)
    }

    result = await db.documents.insert_one(doc_dict)
    doc_id = str(result.inserted_id)

    # Return the initial record. Background processing will update it later.
    updated_doc = await db.documents.find_one({"_id": ObjectId(doc_id)})
    return _format_document(updated_doc), file_path, file_type


def _parse_and_chunk(file_path: Path, file_type: str) -> tuple[str, list[str]]:
    """
    CPU-bound: parse document and split into chunks.
    Runs inside asyncio.to_thread so it doesn't block the event loop.
    """
    text_content = parse_document(file_path, file_type)
    chunks = chunk_text(text_content)
    return text_content, chunks


async def run_document_pipeline_background(
    doc_id: str,
    user_id: str,
    file_path: Path,
    file_type: str,
    original_name: str,
):
    """
    Background task to process the document without blocking the event loop.

    NOTE: We deliberately do NOT accept `db` as a parameter here because
    FastAPI's Depends() generator is torn down after the response is sent.
    Instead we obtain a fresh database handle directly from the client.
    """
    # Import here to avoid circular-import issues
    from app.database import get_db_direct

    db = get_db_direct()

    try:
        logger.info(f"Background pipeline starting for doc={doc_id}, file={original_name}")

        # Step 1: CPU-bound parsing and chunking (run in thread pool)
        text_content, chunks = await asyncio.to_thread(
            _parse_and_chunk, file_path, file_type
        )
        chunk_count = len(chunks)
        logger.info(f"Parsed {chunk_count} chunks from {original_name}")

        # Step 2: Embed and store chunks in vector store
        if chunks:
            vs_manager = VectorStoreManager()
            await vs_manager.add_document_chunks(
                doc_id=doc_id,
                user_id=user_id,
                chunks=chunks,
            )
            logger.info(f"Stored {chunk_count} chunks in vector store for doc={doc_id}")

        # Step 3: Mark document as ready in MongoDB
        await db.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {
                "text_content": text_content,
                "chunk_count": chunk_count,
                "status": "ready",
                "processed_at": datetime.now(timezone.utc),
            }}
        )
        logger.info(f"Document ready: {original_name} ({chunk_count} chunks)")

    except Exception as e:
        logger.error(f"Document processing failed for doc={doc_id}: {e}", exc_info=True)
        try:
            await db.documents.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {
                    "status": "failed",
                    "error_message": str(e),
                    "processed_at": datetime.now(timezone.utc),
                }}
            )
        except Exception as db_err:
            logger.error(f"Failed to update error status in DB: {db_err}")


async def get_user_documents(db: AsyncIOMotorDatabase, user_id: str) -> list[dict]:
    """Get all documents for a user."""
    cursor = db.documents.find({"user_id": user_id}).sort("created_at", -1)
    documents = await cursor.to_list(length=1000)
    return [_format_document(doc) for doc in documents]


async def get_document_by_id(db: AsyncIOMotorDatabase, doc_id: str, user_id: str) -> dict | None:
    """Get a specific document by ID, scoped to user."""
    if not ObjectId.is_valid(doc_id):
        return None
    doc = await db.documents.find_one({"_id": ObjectId(doc_id), "user_id": user_id})
    return _format_document(doc)


async def delete_document(db: AsyncIOMotorDatabase, doc_id: str, user_id: str) -> bool:
    """Delete a document and its vector store data."""
    doc = await get_document_by_id(db, doc_id, user_id)
    if not doc:
        return False

    # Remove from vector store
    vs_manager = VectorStoreManager()
    vs_manager.remove_document(doc_id, user_id)

    # Remove file from disk
    file_path = settings.UPLOAD_DIR / doc["filename"]
    if file_path.exists():
        file_path.unlink()

    # Remove from DB
    await db.documents.delete_one({"_id": ObjectId(doc_id)})
    return True
