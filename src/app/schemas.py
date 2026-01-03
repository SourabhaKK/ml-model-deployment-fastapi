"""
Pydantic models for request/response validation.
CYCLE 2 GREEN PHASE: Input validation via schemas.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    features: List[float] = Field(..., description="Input features for prediction")
    
    @field_validator('features')
    @classmethod
    def features_must_not_be_empty(cls, v: List[float]) -> List[float]:
        """Validate that features list is not empty."""
        if len(v) == 0:
            raise ValueError('features list cannot be empty')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0]
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: float = Field(..., description="Model prediction output")
    model_version: Optional[str] = Field(None, description="Version of the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0.85,
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Health status of the API")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok"
            }
        }
