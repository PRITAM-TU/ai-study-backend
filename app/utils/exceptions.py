"""
Custom exception classes for the application.
"""

from fastapi import HTTPException, status


class AuthenticationError(HTTPException):
    """Raised when authentication fails."""
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class DocumentNotFoundError(HTTPException):
    """Raised when a requested document doesn't exist."""
    def __init__(self, doc_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {doc_id} not found",
        )


class FileProcessingError(HTTPException):
    """Raised when file parsing or processing fails."""
    def __init__(self, detail: str = "Failed to process file"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )


class LLMError(HTTPException):
    """Raised when LLM API call fails."""
    def __init__(self, detail: str = "LLM service unavailable"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


class VectorStoreError(HTTPException):
    """Raised when vector store operations fail."""
    def __init__(self, detail: str = "Vector store operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )
