"""
API route definitions.
CYCLE 4 GREEN PHASE: Error handling for prediction failures.
"""
import asyncio
import logging
import math
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from src.app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.app.dependencies import get_model, DummyModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# F-25: Circuit breaker
# Tracks consecutive prediction failures. After CIRCUIT_BREAKER_THRESHOLD
# failures the circuit opens and all predict requests immediately return 503,
# giving the model time to recover. After CIRCUIT_BREAKER_TIMEOUT seconds it
# moves to HALF-OPEN and allows one trial request through.
#
# Disabled by default (CIRCUIT_BREAKER_ENABLED=false) so tests are unaffected.
# Enable in production: CIRCUIT_BREAKER_ENABLED=true
# ---------------------------------------------------------------------------
_CB_ENABLED: bool = os.getenv("CIRCUIT_BREAKER_ENABLED", "false").lower() == "true"
_CB_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
_CB_TIMEOUT: float = float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30.0"))


class _CircuitBreaker:
    def __init__(self, threshold: int, timeout: float) -> None:
        self.threshold = threshold
        self.timeout = timeout
        self._failures = 0
        self._state = "CLOSED"   # CLOSED | OPEN | HALF_OPEN
        self._opened_at: float = 0.0

    def is_open(self) -> bool:
        if self._state == "OPEN":
            if time.monotonic() - self._opened_at >= self.timeout:
                self._state = "HALF_OPEN"
                logger.info("Circuit breaker → HALF_OPEN: allowing trial request")
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._state = "OPEN"
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker → OPEN after %d consecutive failures", self._failures
            )


_circuit_breaker = _CircuitBreaker(_CB_THRESHOLD, _CB_TIMEOUT)


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

    # F-25: Reject immediately when circuit is open — no point hitting a broken model.
    if _CB_ENABLED and _circuit_breaker.is_open():
        logger.warning("Circuit breaker OPEN — rejecting request", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Try again later.")

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

        # F-21: Explicit cast to Python float — handles numpy scalar / single-element
        # array returned by sklearn/PyTorch without crashing json serialisation.
        prediction = float(prediction)

        # F-04: NaN/Inf from the model would silently crash json.dumps()
        # outside this try block. Validate before building the response.
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite value")

        if _CB_ENABLED:
            _circuit_breaker.record_success()

        # Guard against mock objects / non-string version attributes so that
        # dependency-overridden models in tests don't cause Pydantic rejections.
        raw_version = getattr(model, "version", None)
        model_version = raw_version if isinstance(raw_version, str) else None

        return PredictionResponse(prediction=prediction, model_version=model_version)

    except asyncio.TimeoutError:
        if _CB_ENABLED:
            _circuit_breaker.record_failure()
        logger.error("Prediction timed out", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Prediction failed. Contact support.")

    except (ValueError, RuntimeError) as e:
        # F-12: Log the full error privately; return an opaque message to the
        # caller so internal paths, library versions, and model details are
        # not leaked to potential attackers.
        if _CB_ENABLED:
            _circuit_breaker.record_failure()
        logger.error("Prediction failed", extra={"request_id": request_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="Prediction failed. Contact support.")

    except MemoryError:
        # F-23: MemoryError must NOT be swallowed as a generic HTTP 500.
        # Let it propagate so the OS OOM killer can act and alerting fires.
        logger.critical("OOM during prediction", extra={"request_id": request_id})
        raise

    except Exception as e:
        # Catch-all for any other exception type (e.g. plain Exception from mocks,
        # unexpected library errors). Logs privately and returns a generic 500 so the
        # error is handled within the route handler — not leaked through middleware.
        if _CB_ENABLED:
            _circuit_breaker.record_failure()
        logger.error("Unexpected prediction error", extra={"request_id": request_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="Prediction failed. Contact support.")
