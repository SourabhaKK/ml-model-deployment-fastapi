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
