import pytest
import numpy as np
from fastapi.testclient import TestClient
from api.main import app
from src.inference.predict import create_sample_house

from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

def test_health_endpoint(client):
    """Verify that the health check endpoint returns healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_endpoint_success(client):
    """Verify that the /predict endpoint returns a valid price for a sample house."""
    # Generate sample data
    sample_house = create_sample_house()
    # Convert DataFrame to dictionary for JSON payload
    payload = sample_house.replace({np.nan: None}).to_dict(orient="records")[0]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "currency" in data
    assert data["predicted_price"] > 0

def test_predict_endpoint_invalid_input(client):
    """Verify that the /predict endpoint returns 422 for missing required fields."""
    payload = {"Id": 123} # Missing most fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
