"""
FastAPI application entry point.
GREEN PHASE: Minimal app with /health endpoint wired.
"""
import logging
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.app.config import settings

# F-15: Configure structured logging once at app startup.
# All logger calls across the codebase now emit consistently formatted JSON-like records.
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# F-13: Optional API key authentication.
# Set API_KEY env var to enable. Unset (default) = auth disabled for dev/tests.
# Health and readiness probes are always exempt so load-balancers can reach them.
# ---------------------------------------------------------------------------
_API_KEY: str = settings.api_key  # empty string = disabled

# ---------------------------------------------------------------------------
# F-14: Optional in-memory per-IP rate limiter (sliding window).
# Set RATE_LIMIT_PER_MINUTE to enable (0 = disabled, default for dev/tests).
# For production prefer nginx/envoy upstream rate limiting over this in-process
# store, which does not share state across gunicorn workers (see F-08).
# ---------------------------------------------------------------------------
_RATE_LIMIT: int = settings.rate_limit_per_minute
_rate_store: dict = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    F-07: Eagerly validate that the model loads before accepting traffic.
    If model initialisation fails the app raises immediately — Kubernetes
    marks the pod as unhealthy and never routes requests to it (fail fast).
    F-06: Loading here also eliminates the singleton race condition because
    the model is fully initialised before the first request arrives.
    F-08 NOTE: Each gunicorn worker process runs this lifespan independently,
    creating one model instance per worker (N workers = N × model RAM).
    For large models use a dedicated inference server (Triton/TorchServe) or
    Python multiprocessing.shared_memory so workers share a single copy.
    """
    from src.app.dependencies import get_model
    logger.info("Starting up — loading model")
    get_model()  # raises on failure → pod never becomes ready
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ML Model Deployment API",
    description="Production-grade ML model serving with FastAPI",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# F-09: Global exception handler — catches exceptions that escape route handlers,
# including DI failures from Depends(get_model) which bypass the try/except
# inside predict(). Logs privately with full traceback; returns a generic 500.
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Contact support."},
    )


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    F-13: Enforce X-API-Key header when API_KEY env var is set.
    Health/ready probes are always exempt so orchestrators can reach them.
    Disabled by default (API_KEY unset) so local dev and tests are unaffected.
    """
    exempt = request.url.path in ("/health", "/ready")
    if _API_KEY and not exempt:
        if request.headers.get("X-API-Key") != _API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    F-14: In-memory per-IP sliding-window rate limiter on /predict.
    Set RATE_LIMIT_PER_MINUTE env var to enable (0 = disabled, default).
    WARNING (F-08): this store is process-local — it does not coordinate
    across gunicorn workers. Use nginx/envoy upstream limiting in production.
    """
    if _RATE_LIMIT > 0 and request.url.path == "/predict":
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - 60.0
        # Slide the window: discard timestamps older than 60 s
        _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > cutoff]
        if len(_rate_store[client_ip]) >= _RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        _rate_store[client_ip].append(now)
    return await call_next(request)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    F-15 + F-16: Attach a UUID to every request, log method/path/status/latency,
    and echo the ID back in the X-Request-ID response header so clients can
    correlate their error reports to specific server-side log entries.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    logger.info(
        f"Request started — {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )

    response = await call_next(request)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        f"Request completed — {request.method} {request.url.path} "
        f"{response.status_code} ({latency_ms}ms)",
        extra={"request_id": request_id},
    )

    response.headers["X-Request-ID"] = request_id
    return response


from src.app.routes import router  # noqa: E402 — import after middleware registration
app.include_router(router)
