"""
RAG endpoint: /ask for question-answering with document context.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.rag.service import rag_query

router = APIRouter(prefix="/rag", tags=["RAG Q&A"])


class AskRequest(BaseModel):
    """Schema for the /ask endpoint."""
    question: str = Field(..., min_length=1, max_length=2000)
    doc_id: Optional[int] = None  # Optional: scope to a specific document
    top_k: int = Field(default=5, ge=1, le=20)


class SourceInfo(BaseModel):
    doc_id: int
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    context_used: bool


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Ask a question about your uploaded documents.
    Uses RAG to find relevant context and generate an answer.
    """
    result = await rag_query(
        user_id=current_user.id,
        question=request.question,
        doc_id=request.doc_id,
        top_k=request.top_k,
    )

    return AskResponse(**result)
