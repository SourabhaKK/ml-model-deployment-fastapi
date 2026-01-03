"""
API route definitions.
CYCLE 3 GREEN PHASE: Model dependency injection.
"""
from fastapi import APIRouter, Depends
from src.app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.app.dependencies import get_model, DummyModel

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns the health status of the API.
    """
    return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
async def predict(
    request: PredictionRequest,
    model: DummyModel = Depends(get_model)
) -> PredictionResponse:
    """
    Prediction endpoint with dependency-injected model.
    
    Args:
        request: PredictionRequest with features array
        model: Injected model instance from get_model dependency
        
    Returns:
        PredictionResponse with prediction value from model
    """
    # Use injected model to make prediction
    prediction = model.predict(request.features)
    return PredictionResponse(prediction=prediction)
