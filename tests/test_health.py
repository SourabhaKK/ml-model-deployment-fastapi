"""
Test health endpoint contract.
RED PHASE: These tests MUST FAIL because /health is not implemented yet.
"""
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_health_endpoint_exists():
    """Test that /health endpoint exists and returns 200."""
    response = client.get("/health")
    assert response.status_code == 200, "Health endpoint should return 200 OK"


def test_health_response_structure():
    """Test that /health returns correct JSON structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data, "Response should contain 'status' field"
    assert data["status"] == "ok", "Status should be 'ok'"


def test_health_response_schema():
    """Test that /health response matches HealthResponse schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Validate against expected schema
    assert isinstance(data, dict)
    assert len(data) == 1, "Response should only contain 'status' field"
    assert data.get("status") == "ok"
