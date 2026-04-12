"""
Centralised application configuration via pydantic-settings.
All environment variable access must go through this module.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Model config
    model_version: str = "0.1.0"
    model_path: str = "models/churn_model.pkl"

    # Logging
    log_level: str = "INFO"

    # Auth
    api_key: str = ""

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Circuit breaker
    circuit_breaker_enabled: bool = False

    # Input validation
    expected_feature_count: int = 0  # 0 = disabled


settings = Settings()
