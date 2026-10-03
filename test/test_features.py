import pytest
import pandas as pd
import numpy as np
from src.features.preprocess import identify_features, build_preprocessor, fit_preprocessor

def test_identify_features():
    """Verify that numerical and categorical features are identified correctly."""
    df = pd.DataFrame({
        "Id": [1, 2],
        "SalePrice": [100000, 200000],
        "MSSubClass": [20, 60], # Should be categorical
        "LotArea": [8000, 9000],
        "Street": ["Pave", "Grvl"],
    })

    num, cat = identify_features(df)

    assert "LotArea" in num
    assert "MSSubClass" in cat
    assert "Street" in cat
    assert "Id" not in num
    assert "Id" not in cat
    assert "SalePrice" not in num

def test_fit_preprocessor_returns_transformer():
    """Verify that fit_preprocessor returns a fitted ColumnTransformer."""
    df = pd.DataFrame({
        "Id": [1, 2],
        "LotArea": [8000, 9000],
        "Street": ["Pave", "Grvl"],
        "MSSubClass": [20, 60],
        "MoSold": [1, 5],
    })

    # In a real scenario, we'd drop the target first.
    # fit_preprocessor calls identify_features internally.
    preprocessor = fit_preprocessor(df)

    assert preprocessor is not None
    # Check if it can transform
    transformed = preprocessor.transform(df)
    assert transformed.shape[1] > 0
