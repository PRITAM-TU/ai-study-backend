"""
Pydantic schemas for document endpoints.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class DocumentResponse(BaseModel):
    """Schema for document data in responses."""
    id: str
    filename: str
    original_name: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Schema for listing all user documents."""
    documents: list[DocumentResponse]
    total: int


class UploadResponse(BaseModel):
    """Schema for upload success response."""
    message: str
    document: DocumentResponse
