"""
API route definitions.
GREEN PHASE: Minimal implementation of /health endpoint.
"""
from fastapi import APIRouter
from src.app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns the health status of the API.
    """
    return HealthResponse(status="ok")
