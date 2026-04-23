"""
FastAPI application entry point.
Registers all routers, configures CORS, and initializes the database.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings

# Import all routers
from app.auth.router import router as auth_router
from app.documents.router import router as documents_router
from app.rag.router import router as rag_router
from app.ai_features.router import router as ai_router
from app.audio.router import router as audio_router

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Create directories
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)



    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="🧠 AI-powered study companion with RAG, quiz generation, flashcards, exam mode, and voice features",
    lifespan=lifespan,
)

# Configure CORS
# NOTE: We must list explicit origins (not "*") when sending Authorization headers
# from the frontend. Using "*" with credentials causes browsers to block requests.
ALLOWED_ORIGINS = [
    "https://ai-study-frontend-ochre.vercel.app",  # Production Vercel frontend
    "http://localhost:5173",  # Local Vite dev server
    "http://localhost:3000",  # Local Next.js / CRA dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(ai_router)
app.include_router(audio_router)


# Health check
@app.get("/", tags=["Health"])
async def root():
    return {
        
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
