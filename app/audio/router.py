"""
Audio API endpoints: TTS, Voice Q&A.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.audio.tts_service import text_to_speech, get_available_voices
from app.audio.stt_service import speech_to_text
from app.rag.service import rag_query

router = APIRouter(prefix="/audio", tags=["Audio"])


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice: Optional[str] = None
    rate: Optional[str] = None


class VoiceQAResponse(BaseModel):
    transcription: str
    answer: str
    audio_url: str


@router.post("/tts")
async def create_tts(
    request: TTSRequest,
    current_user: User = Depends(get_current_user),
):
    """Convert text to speech and return the audio file."""
    try:
        audio_path = await text_to_speech(
            text=request.text,
            voice=request.voice,
            rate=request.rate,
        )
        return FileResponse(
            path=str(audio_path),
            media_type="audio/mpeg",
            filename="study_audio.mp3",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@router.get("/voices")
async def list_voices(current_user: User = Depends(get_current_user)):
    """List available TTS voices."""
    try:
        voices = await get_available_voices()
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")


@router.post("/voice-qa")
async def voice_question_answer(
    audio: UploadFile = File(...),
    doc_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
):
    """
    Voice Q&A pipeline:
    1. Transcribe audio question (STT)
    2. Answer using RAG
    3. Convert answer to speech (TTS)
    4. Return transcription + answer + audio URL
    """
    # Step 1: Speech to text
    try:
        audio_bytes = await audio.read()
        transcription = await speech_to_text(audio_bytes, audio.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")

    if not transcription:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    # Step 2: RAG query
    try:
        rag_result = await rag_query(
            user_id=current_user.id,
            question=transcription,
            doc_id=doc_id,
        )
        answer = rag_result["answer"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")

    # Step 3: Answer to speech
    try:
        audio_path = await text_to_speech(answer)
        audio_url = f"/audio/cache/{audio_path.name}"
    except Exception as e:
        audio_url = ""  # TTS failed but we still have the text answer

    return {
        "transcription": transcription,
        "answer": answer,
        "audio_url": audio_url,
    }


@router.get("/cache/{filename}")
async def serve_cached_audio(filename: str):
    """Serve a cached audio file."""
    from app.config import get_settings
    settings = get_settings()
    file_path = settings.AUDIO_CACHE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=str(file_path), media_type="audio/mpeg")
