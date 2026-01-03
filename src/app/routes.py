"""
API route definitions.
CYCLE 1 GREEN PHASE: Implement /health and /predict endpoints.
"""
from fastapi import APIRouter
from src.app.schemas import HealthResponse, PredictionRequest, PredictionResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns the health status of the API.
    """
    return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Prediction endpoint.
    Returns a deterministic dummy prediction (always 0).
    
    Args:
        request: PredictionRequest with features array
        
    Returns:
        PredictionResponse with prediction value
    """
    # Dummy prediction - always return 0
    # No real ML model implementation yet
    return PredictionResponse(prediction=0)
