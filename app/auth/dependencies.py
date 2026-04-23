"""
FastAPI dependencies for authentication.
Provides get_current_user for protected endpoints.
"""

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorDatabase
import jwt

from app.database import get_db
from app.auth.service import decode_access_token, get_user_by_id
from app.utils.exceptions import AuthenticationError

# Security scheme for Swagger UI
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    """
    Dependency: extracts and validates JWT from Authorization header.
    Returns the authenticated user dict or raises 401.
    """
    try:
        token_data = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid or expired token")

    user = await get_user_by_id(db, token_data.user_id)
    if user is None:
        raise AuthenticationError("User not found")
    if not user.get("is_active"):
        raise AuthenticationError("User account is deactivated")

    return user

