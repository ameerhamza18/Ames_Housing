import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.baseline import load_data
from src.features.engineer import engineer_features


TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

RANDOM_STATE = 42
N_SPLITS = 5

ALPHA = 0.0002

SEMANTIC_CATEGORICAL_FEATURES = [
    "MSSubClass",
    "MoSold",
]


def identify_features(df):

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

    if TARGET_COLUMN in numerical_features:
        numerical_features.remove(TARGET_COLUMN)

    if ID_COLUMN in numerical_features:
        numerical_features.remove(ID_COLUMN)

    if ID_COLUMN in categorical_features:
        categorical_features.remove(ID_COLUMN)

    for column in SEMANTIC_CATEGORICAL_FEATURES:

        if column in numerical_features:
            numerical_features.remove(column)

        if column not in categorical_features:
            categorical_features.append(column)

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features,
    categorical_features,
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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


def main():

    print("=" * 80)
    print("AMES HOUSING - LEAKAGE-SAFE CROSS VALIDATION")
    print("=" * 80)

    # ========================================================================
    # 1. Load data
    # ========================================================================

    print("\n[1/5] Loading dataset...")

    df = load_data()

    # ========================================================================
    # 2. Feature engineering
    # ========================================================================

    print("\n[2/5] Applying feature engineering...")

    df = engineer_features(df)

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = np.log1p(
        df[TARGET_COLUMN]
    )

    # ========================================================================
    # 3. Identify features
    # ========================================================================

    print("\n[3/5] Identifying feature types...")

    numerical_features, categorical_features = (
        identify_features(df)
    )

    print(
        f"Numerical features: "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    # ========================================================================
    # 4. Build complete pipeline
    # ========================================================================

    print("\n[4/5] Building ML pipeline...")

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

    # ========================================================================
    # 5. Cross-validation
    # ========================================================================

    print("\n[5/5] Running 5-fold cross-validation...")

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    # Convert negative sklearn error scores
    mae_scores = -results["test_MAE"]
    rmse_scores = -results["test_RMSE"]
    r2_scores = results["test_R2"]

    # ========================================================================
    # Results
    # ========================================================================

    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)

    print("\nNote: CV metrics are calculated on log1p(SalePrice) scale.")

    print("\nMAE:")
    print(
        f"Mean : {mae_scores.mean():.4f}"
    )
    print(
        f"Std  : {mae_scores.std():.4f}"
    )

    print("\nRMSE:")
    print(
        f"Mean : {rmse_scores.mean():.4f}"
    )
    print(
        f"Std  : {rmse_scores.std():.4f}"
    )

    print("\nR²:")
    print(
        f"Mean : {r2_scores.mean():.4f}"
    )
    print(
        f"Std  : {r2_scores.std():.4f}"
    )

    # ========================================================================
    # Individual folds
    # ========================================================================

    fold_results = pd.DataFrame(
        {
            "Fold": range(1, N_SPLITS + 1),
            "MAE": mae_scores,
            "RMSE": rmse_scores,
            "R2": r2_scores,
        }
    )

    print("\n" + "-" * 80)
    print("INDIVIDUAL FOLD RESULTS")
    print("-" * 80)

    print(
        fold_results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()