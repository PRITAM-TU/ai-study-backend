"""
Speech-to-Text service using OpenAI Whisper (tiny model).
Lazy-loads the model to save memory.
"""

import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model reference (lazy loaded)
_model = None


def _get_model():
    """Lazy-load the Whisper model."""
    global _model
    if _model is None:
        import whisper
        from app.config import get_settings
        settings = get_settings()
        logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
        _model = whisper.load_model(settings.WHISPER_MODEL)
        logger.info("Whisper model loaded successfully")
    return _model


async def speech_to_text(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename (for extension detection)

    Returns:
        Transcribed text
    """
    model = _get_model()

    # Write to temp file (Whisper needs a file path)
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="en")
        text = result["text"].strip()
        logger.info(f"STT transcription: {len(text)} chars from {filename}")
        return text
    finally:
        Path(tmp_path).unlink(missing_ok=True)
