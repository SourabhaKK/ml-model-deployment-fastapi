"""
FastAPI application entry point.
GREEN PHASE: Minimal app with /health endpoint wired.
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

# F-15: Configure structured logging once at app startup.
# All logger calls across the codebase now emit consistently formatted JSON-like records.
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    F-07: Eagerly validate that the model loads before accepting traffic.
    If model initialisation fails the app raises immediately — Kubernetes
    marks the pod as unhealthy and never routes requests to it (fail fast).
    F-06: Loading here also eliminates the singleton race condition because
    the model is fully initialised before the first request arrives.
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
        f"Request completed — {request.method} {request.url.path} {response.status_code} ({latency_ms}ms)",
        extra={"request_id": request_id},
    )

    response.headers["X-Request-ID"] = request_id
    return response


from src.app.routes import router  # noqa: E402 — import after middleware registration
app.include_router(router)
