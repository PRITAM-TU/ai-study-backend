"""
Async MongoDB database setup.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import get_settings

settings = get_settings()

# Module-level singleton client — Motor manages the connection pool internally.
# Do NOT recreate this per-request; one instance is correct.
_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return the global Motor client, creating it on first call."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URL)
    return _client


def get_db_direct() -> AsyncIOMotorDatabase:
    """
    Return a database instance directly (not a generator).
    Safe to use in background tasks where FastAPI's dependency injection
    is not available.
    """
    return get_client()[settings.MONGODB_NAME]


async def get_db() -> AsyncIOMotorDatabase:
    """FastAPI dependency: yields an async database instance."""
    yield get_db_direct()
