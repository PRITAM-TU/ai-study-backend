"""
Authentication business logic: password hashing, JWT creation/verification.
"""

from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
import jwt
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from app.config import get_settings
from app.auth.schemas import UserRegister, TokenData

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, email: str) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """Decode and validate a JWT token. Raises jwt.PyJWTError on failure."""
    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )
    return TokenData(user_id=payload["sub"], email=payload["email"])


def _format_user(user: dict | None) -> dict | None:
    if not user:
        return None
    user["id"] = str(user.pop("_id"))
    return user


async def get_user_by_email(db: AsyncIOMotorDatabase, email: str) -> dict | None:
    """Find a user by email address."""
    user = await db.users.find_one({"email": email})
    return _format_user(user)


async def get_user_by_username(db: AsyncIOMotorDatabase, username: str) -> dict | None:
    """Find a user by username."""
    user = await db.users.find_one({"username": username})
    return _format_user(user)


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> dict | None:
    """Find a user by ID."""
    if not ObjectId.is_valid(user_id):
        return None
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    return _format_user(user)


async def register_user(db: AsyncIOMotorDatabase, data: UserRegister) -> dict:
    """Create a new user account."""
    user_dict = {
        "email": data.email,
        "username": data.username,
        "hashed_password": hash_password(data.password),
        "full_name": data.full_name,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "updated_at": None
    }
    result = await db.users.insert_one(user_dict)
    user_dict["_id"] = result.inserted_id
    return _format_user(user_dict)
