"""
Test model dependency injection.
CYCLE 3 RED PHASE: Define model loading and dependency injection behavior.

Requirements:
1. Model should be loaded once at startup (singleton pattern)
2. /predict endpoint should use injected model dependency
3. Model should be accessible via FastAPI dependency injection
"""
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from src.app.main import app
from src.app.dependencies import get_model


client = TestClient(app)

# 51-feature vector matching ChurnModel expected input shape (schema validates this)
FEATURES_51 = [0, 1, 0, 1, 24, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_model_loaded_at_startup():
    """Test that model is loaded once at application startup."""
    # Test that get_model returns a model instance
    model = get_model()
    assert model is not None, "Model should be loaded"

    # Test singleton behavior - same instance returned
    model2 = get_model()
    assert model is model2, "Should return same instance (singleton)"


def test_predict_uses_injected_model():
    """Test that /predict endpoint uses dependency-injected model."""
    # Create a mock model
    mock_model = Mock()
    mock_model.predict.return_value = 0.5

    # Override the dependency
    app.dependency_overrides[get_model] = lambda: mock_model

    try:
        response = client.post(
            "/predict",
            json={"features": FEATURES_51}
        )

        assert response.status_code == 200
        data = response.json()

        # Model's predict method should have been called
        assert mock_model.predict.called, "Model predict should be called"
        assert data["prediction"] == 0.5, "Should return mocked prediction"
    finally:
        # Clean up override
        app.dependency_overrides.clear()


def test_model_predict_called_with_features():
    """Test that model.predict is called with the correct features."""
    # Create a mock model
    mock_model = Mock()
    mock_model.predict.return_value = 0.75

    # Override the dependency
    app.dependency_overrides[get_model] = lambda: mock_model

    try:
        response = client.post(
            "/predict",
            json={"features": FEATURES_51}
        )

        assert response.status_code == 200

        # Verify model.predict was called with features
        mock_model.predict.assert_called_once_with(FEATURES_51)
    finally:
        # Clean up override
        app.dependency_overrides.clear()


def test_model_singleton_behavior():
    """Test that the same model instance is reused across requests."""
    # Make multiple requests without mocking — must use 51 features
    response1 = client.post("/predict", json={"features": FEATURES_51})
    response2 = client.post("/predict", json={"features": FEATURES_51})

    # Both should succeed
    assert response1.status_code == 200
    assert response2.status_code == 200

    # Verify singleton by checking get_model returns same instance
    model1 = get_model()
    model2 = get_model()
    assert model1 is model2, "Should return same model instance (singleton)"
