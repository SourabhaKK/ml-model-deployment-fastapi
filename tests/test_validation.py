"""
Test input validation for /predict endpoint.
CYCLE 2 RED PHASE: Define input validation requirements.

Validation Requirements:
1. Missing features field -> HTTP 422
2. Non-list input for features -> HTTP 422
3. Empty list for features -> HTTP 422 (requires custom validation)
4. Non-numeric features -> HTTP 422
5. Validation errors return FastAPI error structure with 'detail' field

Current Status:
- Tests 1, 2, 4, 5: PASS (Pydantic built-in validation)
- Test 3: FAIL (needs custom validation logic in GREEN phase)
"""
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_predict_rejects_missing_features():
    """Test that /predict returns 422 when features are missing."""
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code == 422, "Should return 422 for missing features"


def test_predict_rejects_invalid_feature_type():
    """Test that /predict returns 422 for invalid feature types."""
    response = client.post(
        "/predict",
        json={"features": "not a list"}
    )
    assert response.status_code == 422, "Should return 422 for invalid feature type"


def test_predict_rejects_non_numeric_features():
    """Test that /predict returns 422 for non-numeric features."""
    response = client.post(
        "/predict",
        json={"features": ["a", "b", "c"]}
    )
    assert response.status_code == 422, "Should return 422 for non-numeric features"


def test_predict_rejects_empty_features():
    """Test that /predict returns 422 for empty feature list."""
    response = client.post(
        "/predict",
        json={"features": []}
    )
    assert response.status_code == 422, "Should return 422 for empty features"


def test_validation_error_response_structure():
    """Test that validation errors return proper FastAPI error structure."""
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data, "Validation error should contain 'detail' field"
    assert isinstance(data["detail"], list), "Detail should be a list of errors"
