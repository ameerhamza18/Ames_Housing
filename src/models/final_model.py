from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.models.baseline import load_data
from src.features.engineer import engineer_features
from src.data.quality import (
    validate_dataset,
    remove_influential_observations,
)


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

TEST_SIZE = 0.20
RANDOM_STATE = 42

ALPHA = 0.0002

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lasso_house_price_model.joblib"
)

SEMANTIC_CATEGORICAL_FEATURES = [
    "MSSubClass",
    "MoSold",
]


# ============================================================================
# FEATURE IDENTIFICATION
# ============================================================================

def identify_features(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:

    numerical_features = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_features = (
        df.select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    # Remove target from input features
    if TARGET_COLUMN in numerical_features:
        numerical_features.remove(TARGET_COLUMN)

    # Remove ID from input features
    if ID_COLUMN in numerical_features:
        numerical_features.remove(ID_COLUMN)

    if ID_COLUMN in categorical_features:
        categorical_features.remove(ID_COLUMN)

    # Treat semantically categorical numerical columns as categorical
    for column in SEMANTIC_CATEGORICAL_FEATURES:

        if column in numerical_features:
            numerical_features.remove(column)

        if column not in categorical_features:
            categorical_features.append(column)

    return numerical_features, categorical_features


# ============================================================================
# PREPROCESSING
# ============================================================================

def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================================
# MODEL PIPELINE
# ============================================================================

def build_model_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                Lasso(
                    alpha=ALPHA,
                    max_iter=100000,
                    tol=1e-4,
                ),
            ),
        ]
    )

    return model


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("AMES HOUSING — FINAL LASSO MODEL")
    print("=" * 80)

    # Initialize MLflow experiment
    mlflow.set_experiment("House-Price-Prediction")

    with mlflow.start_run(run_name="Final-Lasso-Model"):
        # ========================================================================
        # 1. LOAD DATA
        # ========================================================================

        print("\n[1/7] Loading dataset...")

        df = load_data()

        print(
            f"Original dataset shape: {df.shape}"
        )

        # ========================================================================
        # 2. DATA QUALITY
        # ========================================================================

        print("\n[2/7] Validating dataset...")

        validate_dataset(df)

        print(
            "Dataset validation passed."
        )

        df = remove_influential_observations(
            df
        )

        print(
            f"Dataset shape after quality treatment: "
            f"{df.shape}"
        )

        # ========================================================================
        # 3. FEATURE ENGINEERING
        # ========================================================================

        print("\n[3/7] Engineering features...")

        df = engineer_features(
            df
        )

        print(
            f"Feature-engineered shape: {df.shape}"
        )

        # ========================================================================
        # 4. PREPARE FEATURES AND TARGET
        # ========================================================================

        print("\n[4/7] Preparing target and features...")

        X = df.drop(
            columns=[TARGET_COLUMN]
        )

        # Log-transform target
        y = np.log1p(
            df[TARGET_COLUMN]
        )

        print(
            "Target transformation: log1p(SalePrice)"
        )

        # ========================================================================
        # 5. TRAIN TEST SPLIT
        # ========================================================================

        print("\n[5/7] Creating train-test split...")

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
        )

        print(
            f"Training samples: {len(X_train)}"
        )

        print(
            f"Testing samples: {len(X_test)}"
        )

        # ========================================================================
        # 6. BUILD + TRAIN MODEL
        # ========================================================================

        print("\n[6/7] Building and training final model...")

        numerical_features, categorical_features = (
            identify_features(df)
        )

        print(
            f"Numerical features: {len(numerical_features)}"
        )

        print(
            f"Categorical features: {len(categorical_features)}"
        )

        model = build_model_pipeline(
            numerical_features,
            categorical_features,
        )

        print(
            "Model: Lasso Regression"
        )

        print(
            f"Alpha: {ALPHA}"
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Model training completed."
        )

        # Log parameters to MLflow
        mlflow.log_param("alpha", ALPHA)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # ========================================================================
        # SAVE MODEL
        # ========================================================================

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            MODEL_PATH,
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "lasso_model")

        print(
            f"\nModel saved to:\n{MODEL_PATH}"
        )

        # ========================================================================
        # 7. EVALUATION
        # ========================================================================

        print("\n[7/7] Evaluating final model...")

        # Predictions on log scale
        predictions_log = model.predict(
            X_test
        )

        # Convert predictions back to original dollar scale
        predictions = np.expm1(
            predictions_log
        )

        # Convert actual values back to dollar scale
        actual_prices = np.expm1(
            y_test
        )

        # Calculate metrics
        mae = mean_absolute_error(
            actual_prices,
            predictions,
        )

        mse = mean_squared_error(
            actual_prices,
            predictions,
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            actual_prices,
            predictions,
        )

        # Log metrics to MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # ========================================================================
        # RESULTS
        # ========================================================================

        print("\n" + "=" * 80)
        print("FINAL MODEL RESULTS")
        print("=" * 80)

        print(
            f"\nMAE  : ${mae:,.2f}"
        )

        print(
            f"MSE  : ${mse:,.2f}"
        )

        print(
            f"RMSE : ${rmse:,.2f}"
        )

        print(
            f"R²   : {r2:.4f}"
        )

        print("\n" + "=" * 80)
        print("FINAL LASSO MODEL COMPLETED SUCCESSFULLY")
        print("=" * 80)


if __name__ == "__main__":
    main()