"""
End-to-end integration tests.
CYCLE 5 RED PHASE: Full API integration testing with minimal mocking.

These tests verify the complete API flow from request to response,
testing all components working together.
"""
import pytest
from fastapi.testclient import TestClient
from src.app.main import app


client = TestClient(app)

# 51-feature vector matching the churn model's expected input shape
FEATURES_51 = [0, 1, 0, 1, 24, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_full_api_health_check():
    """
    End-to-end test for health check endpoint.
    Tests the complete flow with no mocking.
    """
    response = client.get("/health")

    # Verify response
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    data = response.json()
    assert data == {"status": "ok"}


def test_full_api_prediction_flow():
    """
    End-to-end test for prediction endpoint.
    Tests the complete flow: request → validation → model → response.
    Minimal mocking - uses real app components.
    """
    response = client.post("/predict", json={"features": FEATURES_51})

    # Verify response
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    data = response.json()

    # Verify response structure
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))

    # Real churn model returns 0.0 or 1.0 (binary classification)
    assert data["prediction"] in (0.0, 1.0), "Prediction should be binary churn class"

    # F-05 fix: model_version is now always populated from MODEL_VERSION env var.
    # It should be present and be a non-empty string.
    assert "model_version" in data
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0


def test_full_api_validation_flow():
    """
    End-to-end test for validation error handling.
    Tests that invalid input is properly rejected.
    """
    # Invalid request - empty features list
    request_data = {
        "features": []
    }

    response = client.post("/predict", json=request_data)

    # Verify validation error
    assert response.status_code == 422

    data = response.json()
    assert "detail" in data

    # Verify error details contain validation information
    assert isinstance(data["detail"], list)
    assert len(data["detail"]) > 0


def test_full_api_multiple_requests():
    """
    End-to-end test for multiple sequential requests.
    Verifies that the API handles multiple requests correctly
    and maintains state properly (singleton model).
    """
    # Make multiple prediction requests — all must use 51 features
    for _ in range(3):
        response = client.post("/predict", json={"features": FEATURES_51})

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in (0.0, 1.0)


def test_full_api_health_and_predict():
    """
    End-to-end test combining health check and prediction.
    Verifies both endpoints work together.
    """
    # First check health
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    # Then make a prediction
    predict_response = client.post(
        "/predict",
        json={"features": FEATURES_51}
    )
    assert predict_response.status_code == 200
    assert "prediction" in predict_response.json()

    # Check health again
    health_response2 = client.get("/health")
    assert health_response2.status_code == 200
    assert health_response2.json()["status"] == "ok"


def test_full_api_response_headers():
    """
    End-to-end test verifying response headers are correct.
    """
    # Test health endpoint headers
    health_response = client.get("/health")
    assert "content-type" in health_response.headers
    assert health_response.headers["content-type"] == "application/json"

    # Test predict endpoint headers
    predict_response = client.post(
        "/predict",
        json={"features": FEATURES_51}
    )
    assert "content-type" in predict_response.headers
    assert predict_response.headers["content-type"] == "application/json"
