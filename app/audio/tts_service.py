"""
Text-to-Speech service using edge-tts (Microsoft Edge TTS).
Free, high-quality, no GPU required.
"""

import asyncio
from gtts import gTTS
import hashlib
import logging
from pathlib import Path

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _get_audio_path(text_hash: str) -> Path:
    """Get the file path for a cached audio file."""
    cache_dir = settings.AUDIO_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{text_hash}.mp3"


def _text_hash(text: str) -> str:
    """Create a hash of the text for caching."""
    return hashlib.md5(text.encode()).hexdigest()


async def text_to_speech(
    text: str,
    voice: str | None = None,
    rate: str | None = None,
) -> Path:
    """
    Convert text to speech audio file (MP3).
    Caches results to avoid regenerating the same content.

    Args:
        text: Text to convert to speech
        voice: TTS voice name (default from settings)
        rate: Speech rate adjustment (e.g., "+10%", "-5%")

    Returns:
        Path to the generated MP3 file
    """
    voice = voice or settings.TTS_VOICE
    rate = rate or settings.TTS_RATE

    # Check cache
    cache_key = _text_hash(f"{text}_{voice}_{rate}")
    audio_path = _get_audio_path(cache_key)

    if audio_path.exists():
        logger.info(f"TTS cache hit: {cache_key}")
        return audio_path

    # Generate audio
    logger.info(f"Generating TTS audio ({len(text)} chars)")
    
    def _generate_and_save():
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(str(audio_path))
        
    await asyncio.to_thread(_generate_and_save)

    logger.info(f"TTS audio saved: {audio_path}")
    return audio_path


async def get_available_voices() -> list[dict]:
    """Get list of available TTS voices. (Mocked for gTTS)"""
    return [
        {
            "name": "Default Voice",
            "gender": "Female",
            "locale": "en-US",
        }
    ]
