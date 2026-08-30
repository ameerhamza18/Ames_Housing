import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.inference.predict import create_sample_house

client = TestClient(app)

def test_health_endpoint():
    """Verify that the health check endpoint returns healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_endpoint_success():
    """Verify that the /predict endpoint returns a valid price for a sample house."""
    # Generate sample data
    sample_house = create_sample_house()
    # Convert DataFrame to dictionary for JSON payload
    payload = sample_house.to_dict(orient="records")[0]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "currency" in data
    assert data["predicted_price"] > 0

def test_predict_endpoint_invalid_input():
    """Verify that the /predict endpoint returns 422 for missing required fields."""
    payload = {"Id": 123} # Missing most fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
