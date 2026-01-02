"""
Application configuration.
RED PHASE: Basic config structure only.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    app_name: str = "ML Model Deployment API"
    debug: bool = False
    model_path: str = "models/model.pkl"
    
    class Config:
        env_file = ".env"


settings = Settings()
