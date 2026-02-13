"""
Structured error handling with error codes and request tracking.
"""

import logging
import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# ============================================================================
# ERROR CODES ENUM
# ============================================================================


class ErrorCode(str, Enum):
    """Standardized error codes for the API."""

    # Client errors (4xx)
    INVALID_INPUT = "INVALID_INPUT"
    TICKER_INVALID = "TICKER_INVALID"
    TICKER_NOT_FOUND = "TICKER_NOT_FOUND"
    TICKER_NOT_SUPPORTED = "TICKER_NOT_SUPPORTED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    TRAINING_ERROR = "TRAINING_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class APIValidationError(APIError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
        )


class TickerError(APIError):
    """Raised for ticker-related errors."""

    def __init__(
        self,
        message: str,
        ticker: str,
        error_code: ErrorCode = ErrorCode.TICKER_INVALID,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=400,
            details={"ticker": ticker},
        )


class PredictionError(APIError):
    """Raised when prediction generation fails."""

    def __init__(
        self,
        message: str,
        ticker: str,
        details: Optional[Dict] = None,
    ):
        d = details or {}
        d["ticker"] = ticker
        super().__init__(
            message=message,
            error_code=ErrorCode.PREDICTION_ERROR,
            status_code=500,
            details=d,
        )


class DatabaseError(APIError):
    """Raised when database operations fail."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=503,
            details=details,
        )


# ============================================================================
# ERROR RESPONSE BUILDER
# ============================================================================


class ErrorResponse:
    """Build structured error responses."""

    @staticmethod
    def build(
        error_code: ErrorCode,
        message: str,
        status_code: int,
        request: Request,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a structured error response."""
        return {
            "error": {
                "code": error_code.value,
                "message": message,
                "details": details or {},
                "request_id": request_id or str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path,
                "status": status_code,
            }
        }

    @staticmethod
    def from_exception(
        exc: Exception,
        request: Request,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build error response from exception."""
        if isinstance(exc, APIError):
            return ErrorResponse.build(
                error_code=exc.error_code,
                message=exc.message,
                status_code=exc.status_code,
                request=request,
                details=exc.details,
                request_id=request_id,
            )
        elif isinstance(exc, StarletteHTTPException):
            detail = exc.detail
            if isinstance(detail, dict):
                message = detail.get("message", str(detail))
            else:
                message = str(detail)
            return ErrorResponse.build(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=message,
                status_code=exc.status_code,
                request=request,
                request_id=request_id,
            )
        else:
            return ErrorResponse.build(
                error_code=ErrorCode.INTERNAL_ERROR,
                message="An unexpected error occurred",
                status_code=500,
                request=request,
                details={"error_type": type(exc).__name__},
                request_id=request_id,
            )


# ============================================================================
# MIDDLEWARE FOR REQUEST TRACKING
# ============================================================================


from starlette.middleware.base import BaseHTTPMiddleware


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests and responses."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        logger.info(
            "Request started: %s %s",
            request.method,
            request.url.path,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.error(
        "API error: %s",
        exc.error_code,
        extra={
            "request_id": request_id,
            "error_code": exc.error_code.value,
            "message": exc.message,
            "details": exc.details,
            "status_code": exc.status_code,
        },
    )

    body = ErrorResponse.from_exception(exc, request, request_id)
    return JSONResponse(status_code=exc.status_code, content=body)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(
        "Validation error",
        extra={"request_id": request_id, "errors": errors},
    )

    body = ErrorResponse.build(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        status_code=422,
        request=request,
        details={"validation_errors": errors},
        request_id=request_id,
    )

    return JSONResponse(status_code=422, content=body)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    error_code_map = {
        400: ErrorCode.INVALID_INPUT,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }
    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    detail = exc.detail
    message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)

    logger.error(
        "HTTP exception: %d",
        exc.status_code,
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )

    body = ErrorResponse.build(
        error_code=error_code,
        message=message,
        status_code=exc.status_code,
        request=request,
        request_id=request_id,
    )

    return JSONResponse(status_code=exc.status_code, content=body)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.error(
        "Unexpected error",
        extra={
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )

    body = ErrorResponse.build(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred. Please try again later.",
        status_code=500,
        request=request,
        details={"error_id": request_id},
        request_id=request_id,
    )

    return JSONResponse(status_code=500, content=body)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def raise_ticker_error(ticker: str, message: str):
    """Raise a ticker error with consistent format."""
    raise TickerError(message=message, ticker=ticker)


def raise_validation_error(field: str, message: str):
    """Raise a validation error with consistent format."""
    raise APIValidationError(
        message=f"Validation failed for {field}",
        details={"field": field, "reason": message},
    )


# ============================================================================
# SETUP FUNCTION
# ============================================================================


def setup_error_handling(app):
    """
    Set up all error handlers for the FastAPI app.

    Call this in main.py after creating the app: setup_error_handling(app)
    """
    app.add_middleware(RequestTrackingMiddleware)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Error handling configured")
