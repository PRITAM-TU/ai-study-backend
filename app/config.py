"""
Application configuration loaded from environment variables.
Uses pydantic-settings for validation and .env file support.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    """All application settings, loaded from .env file."""

    # ── App ──
    APP_NAME: str = "AI Study Companion"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── JWT Auth ──
    JWT_SECRET_KEY: str = "change-this-to-a-random-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ── Database ──
    MONGODB_URL: str = "mongodb+srv://pritamtung03_db_user:LfkdyeIcdwKVD9dh@cluster0.4ahq9pc.mongodb.net/"
    MONGODB_NAME: str = "ai_study_db"

    # ── LLM (Ollama) ──
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = "ollama"  # Ollama doesn't need a real key
    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    # ── Embeddings ──
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ── RAG ──
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 5

    # ── File Storage ──
    UPLOAD_DIR: Path = Path("uploads")
    AUDIO_CACHE_DIR: Path = Path("audio_cache")
    VECTOR_STORE_DIR: Path = Path("vector_stores")
    MAX_UPLOAD_SIZE_MB: int = 50

    # ── TTS ──
    TTS_VOICE: str = "en-US-AriaNeural"
    TTS_RATE: str = "+0%"

    # ── Whisper STT ──
    WHISPER_MODEL: str = "tiny"

    # ── CORS ──
    CORS_ORIGINS: list[str] = ["https://ai-study-frontend-ochre.vercel.app", "https://ai-study-backend-4gl8.onrender.com"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance (singleton)."""
    return Settings()
