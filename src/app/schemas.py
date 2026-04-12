"""
Pydantic models for request/response validation.
CYCLE 2 GREEN PHASE: Input validation via schemas.
"""
import os
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional

# F-02: Set EXPECTED_FEATURE_COUNT to enforce the model's exact input shape.
# Defaults to 0 (disabled) so DummyModel and tests are unaffected.
# In production set: EXPECTED_FEATURE_COUNT=512 (or whatever your model expects).
_EXPECTED_FEATURE_COUNT: int = int(os.getenv("EXPECTED_FEATURE_COUNT", "0"))


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    # F-01: max_length=1000 prevents OOM from arbitrarily large payloads
    features: List[float] = Field(..., description="Input features for prediction", max_length=1000)

    @field_validator('features')
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        """Validate that features list is non-empty and matches expected shape."""
        if len(v) == 0:
            raise ValueError('features list cannot be empty')
        # F-02: Enforce model input shape when configured
        if _EXPECTED_FEATURE_COUNT > 0 and len(v) != _EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f'Expected {_EXPECTED_FEATURE_COUNT} features, got {len(v)}'
            )
        return v

    # F-26: ConfigDict replaces deprecated class Config (Pydantic v2)
    # F-03: strict=True rejects silent coercion — "1.5" (str) is rejected instead of
    # being silently cast to 1.5 (float), surfacing misconfigured clients early.
    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0]
            }
        }
    )


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
