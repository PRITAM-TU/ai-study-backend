"""
Async MongoDB database setup.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import get_settings

settings = get_settings()

client = AsyncIOMotorClient(settings.MONGODB_URL)

async def get_db() -> AsyncIOMotorDatabase:
    """Dependency: yields an async database instance."""
    db = client[settings.MONGODB_NAME]
    yield db

