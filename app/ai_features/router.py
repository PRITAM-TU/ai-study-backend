"""
AI Features API endpoints: summary, quiz, flashcards, exam-mode.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.documents.service import get_document_by_id
from app.rag.vectorstore import VectorStoreManager
from app.ai_features.schemas import (
    SummaryRequest, SummaryResponse,
    QuizRequest, QuizResponse, QuizSubmitRequest, QuizResult,
    FlashcardRequest, FlashcardResponse,
    ExamRequest, ExamPrediction,
)
from app.ai_features.summary import generate_summary
from app.ai_features.quiz import generate_quiz, score_quiz
from app.ai_features.flashcards import generate_flashcards
from app.ai_features.exam_mode import predict_exam, generate_lazy_mode_script

router = APIRouter(prefix="/ai", tags=["AI Features"])


async def _get_document_text(doc_id: int, user_id: int, db: AsyncSession) -> str:
    """Helper: retrieve document text from vector store or DB."""
    doc = await get_document_by_id(db, doc_id, user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document is still processing")

    # Get text from vector store chunks (reconstructed)
    vs = VectorStoreManager()
    text = vs.get_document_text(user_id, doc_id)

    if not text:
        # Fallback to stored text content
        text = doc.text_content or ""

    if not text:
        raise HTTPException(status_code=400, detail="No text content found for this document")

    return text


# ── Summary ──
@router.post("/summary", response_model=SummaryResponse)
async def create_summary(
    request: SummaryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a structured summary of a document."""
    text = await _get_document_text(request.doc_id, current_user.id, db)

    try:
        result = await generate_summary(text)
        return SummaryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


# ── Quiz ──
@router.post("/quiz", response_model=QuizResponse)
async def create_quiz(
    request: QuizRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a quiz from a document."""
    text = await _get_document_text(request.doc_id, current_user.id, db)

    try:
        result = await generate_quiz(text, request.num_questions, request.difficulty)
        return QuizResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")


@router.post("/quiz/score", response_model=QuizResult)
async def submit_quiz(
    request: QuizSubmitRequest,
    current_user: User = Depends(get_current_user),
):
    """Score a quiz submission."""
    questions = [q.model_dump() for q in request.quiz.questions]
    result = score_quiz(questions, request.answers)
    return QuizResult(**result)


# ── Flashcards ──
@router.post("/flashcards", response_model=FlashcardResponse)
async def create_flashcards(
    request: FlashcardRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate flashcards from a document."""
    text = await _get_document_text(request.doc_id, current_user.id, db)

    try:
        result = await generate_flashcards(text, request.num_cards)
        return FlashcardResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")


# ── Exam Mode ──
@router.post("/exam-mode", response_model=ExamPrediction)
async def create_exam_prediction(
    request: ExamRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Predict exam questions and identify important topics."""
    text = await _get_document_text(request.doc_id, current_user.id, db)

    try:
        result = await predict_exam(text)
        return ExamPrediction(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exam prediction failed: {str(e)}")


# ── Lazy Mode ──
@router.post("/lazy-mode")
async def create_lazy_mode(
    request: ExamRequest,  # Reuse: just needs doc_id
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a TTS-friendly audio script from study material."""
    text = await _get_document_text(request.doc_id, current_user.id, db)

    try:
        result = await generate_lazy_mode_script(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lazy mode generation failed: {str(e)}")
