"""
Document upload and management endpoints.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.documents.schemas import DocumentResponse, DocumentListResponse, UploadResponse
from app.documents.service import process_document, run_document_pipeline_background, get_user_documents, delete_document
from app.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a PDF, PPTX, or TXT file for processing.
    The file is parsed, chunked, embedded, and stored in the vector database.
    """
    # Validate file type
    allowed_types = {"application/pdf", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "text/plain"}
    if file.content_type not in allowed_types and not file.filename.endswith((".pdf", ".pptx", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Allowed: PDF, PPTX, TXT",
        )

    # Validate file size
    content = await file.read()
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    try:
        doc, file_path, file_type = await process_document(db, current_user["id"], content, file.filename)
        
        # Schedule the AI processing in the background
        background_tasks.add_task(
            run_document_pipeline_background,
            db=db,
            doc_id=doc["id"],
            user_id=current_user["id"],
            file_path=file_path,
            file_type=file_type,
            original_name=file.filename
        )

        return UploadResponse(
            message="Document uploaded and processing started in background",
            document=DocumentResponse.model_validate(doc),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List all documents for the current user."""
    docs = await get_user_documents(db, current_user["id"])
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
    )


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(
    doc_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Delete a document and its vector embeddings."""
    deleted = await delete_document(db, doc_id, current_user["id"])
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

