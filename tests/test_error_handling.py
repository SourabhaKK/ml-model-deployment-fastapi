"""
Test prediction failure handling.
CYCLE 4 RED PHASE: Define error handling for prediction-time failures.

Requirements:
1. Model throws exception → HTTP 500 with meaningful error message
2. Invalid prediction output → HTTP 500 with meaningful error message
3. Error responses should include error details for debugging

These tests MUST FAIL because error handling is not implemented yet.
"""
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from src.app.main import app
from src.app.dependencies import get_model


client = TestClient(app)


def test_model_exception_returns_500():
    """Test that model exceptions return HTTP 500 with error message."""
    # Create a mock model that raises an exception
    mock_model = Mock()
    mock_model.predict.side_effect = Exception("Model prediction failed")
    
    # Override the dependency
    app.dependency_overrides[get_model] = lambda: mock_model
    
    try:
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0]}
        )
        
        # Should return 500 for internal server error
        assert response.status_code == 500, "Should return 500 for model exception"
        
        # Response should contain error information
        data = response.json()
        assert "detail" in data or "error" in data, "Should include error details"
    finally:
        app.dependency_overrides.clear()


def test_model_exception_includes_error_message():
    """Test that error response includes meaningful error message."""
    # Create a mock model that raises a specific exception
    mock_model = Mock()
    error_message = "Unable to process features: invalid shape"
    mock_model.predict.side_effect = ValueError(error_message)
    
    # Override the dependency
    app.dependency_overrides[get_model] = lambda: mock_model
    
    try:
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0]}
        )
        
        assert response.status_code == 500
        data = response.json()
        
        # Error message should be included in response
        response_text = str(data)
        assert "error" in response_text.lower() or "detail" in response_text.lower(), \
            "Response should indicate an error occurred"
    finally:
        app.dependency_overrides.clear()


def test_runtime_error_during_prediction():
    """Test that runtime errors during prediction are handled gracefully."""
    # Create a mock model that raises RuntimeError
    mock_model = Mock()
    mock_model.predict.side_effect = RuntimeError("Model runtime error")
    
    # Override the dependency
    app.dependency_overrides[get_model] = lambda: mock_model
    
    try:
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0, 4.0]}
        )
        
        # Should return 500 for runtime error
        assert response.status_code == 500, "Should return 500 for runtime error"
        
        # Should return JSON response (not plain text error)
        assert response.headers["content-type"] == "application/json", \
            "Should return JSON error response"
    finally:
        app.dependency_overrides.clear()


def test_prediction_continues_after_error():
    """Test that API continues to work after a prediction error."""
    # Create a mock model that fails once then succeeds
    mock_model = Mock()
    mock_model.predict.side_effect = [
        Exception("Temporary failure"),
        0.5  # Success on second call
    ]
    
    # Override the dependency
    app.dependency_overrides[get_model] = lambda: mock_model
    
    try:
        # First request should fail
        response1 = client.post(
            "/predict",
            json={"features": [1.0, 2.0]}
        )
        assert response1.status_code == 500, "First request should fail"
        
        # Second request should succeed
        response2 = client.post(
            "/predict",
            json={"features": [3.0, 4.0]}
        )
        assert response2.status_code == 200, "Second request should succeed"
        assert response2.json()["prediction"] == 0.5
    finally:
        app.dependency_overrides.clear()
