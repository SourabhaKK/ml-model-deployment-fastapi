"""
Pydantic models for request/response validation.
CYCLE 2 GREEN PHASE: Input validation via schemas.
"""
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    # F-01: max_length=1000 prevents OOM from arbitrarily large payloads
    features: List[float] = Field(..., description="Input features for prediction", max_length=1000)

    @field_validator('features')
    @classmethod
    def features_must_not_be_empty(cls, v: List[float]) -> List[float]:
        """Validate that features list is not empty."""
        if len(v) == 0:
            raise ValueError('features list cannot be empty')
        return v

    # F-26: ConfigDict replaces deprecated class Config (Pydantic v2)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "features": [1.0, 2.0, 3.0, 4.0]
        }
    })


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: float = Field(..., description="Model prediction output")
    model_version: Optional[str] = Field(None, description="Version of the model used")

    # F-26: ConfigDict replaces deprecated class Config (Pydantic v2)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prediction": 0.85,
            "model_version": "1.0.0"
        }
    })


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Health status of the API")

    # F-26: ConfigDict replaces deprecated class Config (Pydantic v2)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "ok"
        }
    })
