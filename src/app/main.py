"""
FastAPI application entry point.
GREEN PHASE: Minimal app with /health endpoint wired.
"""
from fastapi import FastAPI
from src.app.routes import router

app = FastAPI(
    title="ML Model Deployment API",
    description="Production-grade ML model serving with FastAPI",
    version="0.1.0"
)

# Include routes
app.include_router(router)
