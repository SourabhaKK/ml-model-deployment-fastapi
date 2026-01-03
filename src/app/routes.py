"""
API route definitions.
CYCLE 4 GREEN PHASE: Error handling for prediction failures.
"""
from fastapi import APIRouter, Depends, HTTPException
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
    Prediction endpoint with dependency-injected model and error handling.
    
    Args:
        request: PredictionRequest with features array
        model: Injected model instance from get_model dependency
        
    Returns:
        PredictionResponse with prediction value from model
        
    Raises:
        HTTPException: 500 if prediction fails
    """
    try:
        # Use injected model to make prediction
        prediction = model.predict(request.features)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        # Catch any exception during prediction and return HTTP 500
        error_message = f"Prediction failed: {str(e)}"
        raise HTTPException(
            status_code=500,
            detail=error_message
        )
