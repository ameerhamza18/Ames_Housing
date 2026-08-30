from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.engineer import engineer_features


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lasso_house_price_model.joblib"
)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the complete trained ML pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================================
# PREDICTION
# ============================================================================

def predict_price(
    model,
    house_data: pd.DataFrame,
) -> float:
    """
    Predict house price from raw house features.

    Feature engineering is applied before prediction.
    """

    # Apply the same feature engineering used during training
    engineered_data = engineer_features(
        house_data
    )

    # Model was trained on log1p(SalePrice)
    predicted_log_price = model.predict(
        engineered_data
    )

    # Convert back to original dollar scale
    predicted_price = np.expm1(
        predicted_log_price
    )

    return float(predicted_price[0])


# ============================================================================
# SAMPLE HOUSE
# ============================================================================

def create_sample_house() -> pd.DataFrame:
    """
    Create a new house observation.

    This represents NEW input data rather than
    loading an existing SalePrice from the dataset.
    """

    house = {
        "Id": 9999,

        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotFrontage": 70,
        "LotArea": 8500,

        "Street": "Pave",
        "Alley": np.nan,

        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",

        "Neighborhood": "CollgCr",
        "Condition1": "Norm",
        "Condition2": "Norm",

        "BldgType": "1Fam",
        "HouseStyle": "2Story",

        "OverallQual": 7,
        "OverallCond": 5,

        "YearBuilt": 2005,
        "YearRemodAdd": 2005,

        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",

        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",

        "MasVnrType": "BrkFace",
        "MasVnrArea": 100,

        "ExterQual": "Gd",
        "ExterCond": "TA",

        "Foundation": "PConc",

        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "Gd",
        "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 600,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 400,
        "TotalBsmtSF": 1000,

        "Heating": "GasA",
        "HeatingQC": "Ex",

        "CentralAir": "Y",

        "Electrical": "SBrkr",

        "1stFlrSF": 1000,
        "2ndFlrSF": 800,

        "LowQualFinSF": 0,
        "GrLivArea": 1800,

        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,

        "FullBath": 2,
        "HalfBath": 1,

        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,

        "KitchenQual": "Gd",
        "TotRmsAbvGrd": 7,

        "Functional": "Typ",

        "Fireplaces": 1,
        "FireplaceQu": "Gd",

        "GarageType": "Attchd",
        "GarageYrBlt": 2005,
        "GarageFinish": "Fin",
        "GarageCars": 2,
        "GarageArea": 500,
        "GarageQual": "TA",
        "GarageCond": "TA",

        "PavedDrive": "Y",

        "WoodDeckSF": 100,
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,

        "PoolArea": 0,
        "PoolQC": np.nan,

        "Fence": np.nan,
        "MiscFeature": np.nan,
        "MiscVal": 0,

        "MoSold": 6,
        "YrSold": 2008,
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }

    return pd.DataFrame([house])


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("AMES HOUSING — NEW HOUSE PREDICTION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------------

    print("\n[1/4] Loading trained model...")

    model = load_model()

    print("Model loaded successfully.")

    # ------------------------------------------------------------------------
    # 2. Create new house
    # ------------------------------------------------------------------------

    print("\n[2/4] Creating new house input...")

    house = create_sample_house()

    print(
        f"Raw input shape: {house.shape}"
    )

    # ------------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------------

    print("\n[3/4] Applying feature engineering...")

    engineered_house = engineer_features(
        house
    )

    print(
        f"Engineered input shape: "
        f"{engineered_house.shape}"
    )

    # ------------------------------------------------------------------------
    # 4. Prediction
    # ------------------------------------------------------------------------

    print("\n[4/4] Generating prediction...")

    predicted_price = predict_price(
        model,
        house,
    )

    print("\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)

    print(
        f"\nPredicted SalePrice: "
        f"${predicted_price:,.2f}"
    )

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()