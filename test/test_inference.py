import pytest
import pandas as pd
import numpy as np
from src.inference.predict import load_model, predict_price, create_sample_house

@pytest.fixture(scope="module")
def model():
    """Fixture to load the model once for all tests in the module."""
    return load_model()

def test_predict_price_returns_float(model):
    """Verify that predict_price returns a float."""
    house = create_sample_house()
    price = predict_price(model, house)
    assert isinstance(price, float)
    assert price > 0

def test_predict_price_consistent(model):
    """Verify that the same input produces the same prediction."""
    house = create_sample_house()
    price1 = predict_price(model, house)
    price2 = predict_price(model, house)
    assert price1 == price2

def test_predict_price_with_modified_input(model):
    """Verify that changing house features changes the prediction."""
    house1 = create_sample_house()
    price1 = predict_price(model, house1)

    # Increase OverallQual to significantly increase price
    house2 = house1.copy()
    house2.at[0, "OverallQual"] = 10
    price2 = predict_price(model, house2)

    assert price2 != price1
