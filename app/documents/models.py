"""
SQLAlchemy Document model for uploaded study materials.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class Document(Base):
    """Uploaded document metadata."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    original_name = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)  # pdf, pptx, txt
    file_size = Column(Integer, nullable=False)  # bytes
    text_content = Column(Text, nullable=True)  # extracted text
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="processing")  # processing, ready, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Document(id={self.id}, name={self.original_name}, status={self.status})>"
