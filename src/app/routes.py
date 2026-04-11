"""
API route definitions.
CYCLE 4 GREEN PHASE: Error handling for prediction failures.
"""
import asyncio
import logging
import math

from fastapi import APIRouter, Depends, HTTPException, Request
from src.app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.app.dependencies import get_model, DummyModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe — returns ok if the process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(model: DummyModel = Depends(get_model)) -> HealthResponse:
    """
    F-17: Readiness probe — returns ok only when the model is loaded.
    Configure Kubernetes readinessProbe on /ready, not /health.
    Returns 503 if the model is not available so Kubernetes stops routing traffic.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return HealthResponse(status="ready")


@router.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
async def predict(
    request: PredictionRequest,
    http_request: Request,
    model: DummyModel = Depends(get_model)
) -> PredictionResponse:
    """
    Prediction endpoint with dependency-injected model and error handling.

    Args:
        request: PredictionRequest with features array
        http_request: FastAPI Request (used for request_id logging)
        model: Injected model instance from get_model dependency

    Returns:
        PredictionResponse with prediction value and model version

    Raises:
        HTTPException: 500 if prediction fails or times out
    """
    request_id = getattr(http_request.state, "request_id", "unknown")

    try:
        loop = asyncio.get_event_loop()

        # F-10: Offload synchronous CPU-bound inference to a thread pool so
        # the asyncio event loop is not blocked during model.predict().
        # F-11: Enforce a 5-second hard timeout — hung inference never ties
        # up the worker indefinitely.
        prediction = await asyncio.wait_for(
            loop.run_in_executor(None, model.predict, request.features),
            timeout=5.0
        )

        # F-04: NaN/Inf from the model would silently crash json.dumps()
        # outside this try block. Validate before building the response.
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite value")

        return PredictionResponse(prediction=prediction, model_version=model.version)

    except asyncio.TimeoutError:
        logger.error("Prediction timed out", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Prediction failed. Contact support.")

    except (ValueError, RuntimeError) as e:
        # F-12: Log the full error privately; return an opaque message to the
        # caller so internal paths, library versions, and model details are
        # not leaked to potential attackers.
        logger.error("Prediction failed", extra={"request_id": request_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="Prediction failed. Contact support.")

    except MemoryError:
        # F-23: MemoryError must NOT be swallowed as a generic HTTP 500.
        # Let it propagate so the OS OOM killer can act and alerting fires.
        logger.critical("OOM during prediction", extra={"request_id": request_id})
        raise
