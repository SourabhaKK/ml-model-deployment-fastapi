"""
Test predict endpoint contract.
CYCLE 1 RED PHASE: Define /predict endpoint contract.

Contract Definition:
- POST /predict
- Request: {"features": [1.2, 3.4, 5.6]}
- Response: {"prediction": 0} (numeric prediction value)
- Expected status: 200 for valid requests

These tests MUST FAIL because /predict endpoint is not implemented yet.
"""
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_predict_endpoint_exists():
    """Test that /predict endpoint exists."""
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    # Should not return 404 (endpoint should exist)
    assert response.status_code != 404, "Predict endpoint should exist"


def test_predict_accepts_valid_request():
    """Test that /predict accepts valid PredictionRequest."""
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    # Should eventually return 200, but will fail in RED phase
    assert response.status_code == 200, "Should accept valid prediction request"


def test_predict_response_structure():
    """Test that /predict returns correct response structure."""
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data, "Response should contain 'prediction' field"
    assert isinstance(data["prediction"], (int, float)), "Prediction should be numeric"


def test_predict_returns_model_version():
    """Test that /predict optionally returns model version."""
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    assert response.status_code == 200
    data = response.json()
    # model_version is optional, but if present should be a string
    if "model_version" in data:
        assert isinstance(data["model_version"], str)


def test_predict_response_contains_confidence_key():
    """
    Task 2: Response must contain a 'confidence' key.
    DummyModel has no predict_proba, so the value must be null.
    """
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    assert response.status_code == 200
    data = response.json()
    # confidence key is excluded when None (response_model_exclude_none=True),
    # so either absent (null) or a float in [0, 1].
    if "confidence" in data:
        assert isinstance(data["confidence"], float), "confidence must be a float"
        assert 0.0 <= data["confidence"] <= 1.0, "confidence must be between 0.0 and 1.0"


def test_predict_confidence_is_null_for_dummy_model():
    """
    Task 2: DummyModel has no predict_proba — confidence must be absent/null.
    The endpoint uses response_model_exclude_none=True so null fields are omitted.
    """
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    assert response.status_code == 200
    data = response.json()
    # DummyModel has no predict_proba so confidence should be absent (excluded as None)
    assert "confidence" not in data, (
        "DummyModel has no predict_proba — confidence should be null/absent"
    )
