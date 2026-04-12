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
    # Valid prediction request
    request_data = {
        "features": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    
    response = client.post("/predict", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    
    data = response.json()
    
    # Verify response structure
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    
    # Verify prediction value (dummy model returns 0.0)
    assert data["prediction"] == 0.0

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
    # Make multiple prediction requests
    requests = [
        {"features": [1.0, 2.0]},
        {"features": [3.0, 4.0, 5.0]},
        {"features": [6.0, 7.0, 8.0, 9.0]},
    ]
    
    for request_data in requests:
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.0


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
        json={"features": [1.0, 2.0, 3.0]}
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
        json={"features": [1.0, 2.0]}
    )
    assert "content-type" in predict_response.headers
    assert predict_response.headers["content-type"] == "application/json"
