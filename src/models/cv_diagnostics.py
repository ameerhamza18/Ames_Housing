import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_predict
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

    return ColumnTransformer(
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


def main():

    print("=" * 80)
    print("AMES HOUSING - CROSS VALIDATION DIAGNOSTICS")
    print("=" * 80)

    # ========================================================================
    # Load + engineer
    # ========================================================================

    df = load_data()

    df = engineer_features(df)

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y_original = df[TARGET_COLUMN]

    y_log = np.log1p(
        y_original
    )

    # ========================================================================
    # Features
    # ========================================================================

    numerical_features, categorical_features = (
        identify_features(df)
    )

    # ========================================================================
    # Pipeline
    # ========================================================================

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
    # CV
    # ========================================================================

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\nGenerating out-of-fold predictions...")

    predictions_log = cross_val_predict(
        model,
        X,
        y_log,
        cv=cv,
        n_jobs=-1,
    )

    predictions = np.expm1(
        predictions_log
    )

    # ========================================================================
    # Diagnostic dataframe
    # ========================================================================

    diagnostics = pd.DataFrame(
        {
            "Id": X[ID_COLUMN].values,
            "ActualPrice": y_original.values,
            "PredictedPrice": predictions,
        }
    )

    diagnostics["Residual"] = (
        diagnostics["ActualPrice"]
        - diagnostics["PredictedPrice"]
    )

    diagnostics["AbsoluteError"] = (
        diagnostics["Residual"].abs()
    )

    diagnostics["APE"] = (
        diagnostics["AbsoluteError"]
        / diagnostics["ActualPrice"]
        * 100
    )

    # ========================================================================
    # Largest errors
    # ========================================================================

    print("\n" + "=" * 80)
    print("20 LARGEST CROSS-VALIDATION ERRORS")
    print("=" * 80)

    largest_errors = (
        diagnostics
        .sort_values(
            "AbsoluteError",
            ascending=False,
        )
        .head(20)
    )

    print(
        largest_errors[
            [
                "Id",
                "ActualPrice",
                "PredictedPrice",
                "Residual",
                "AbsoluteError",
                "APE",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================================
    # Error statistics
    # ========================================================================

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION ERROR STATISTICS")
    print("=" * 80)

    print(
        f"\nMean Absolute Error : "
        f"${diagnostics['AbsoluteError'].mean():,.2f}"
    )

    print(
        f"Median Absolute Error : "
        f"${diagnostics['AbsoluteError'].median():,.2f}"
    )

    print(
        f"Maximum Absolute Error : "
        f"${diagnostics['AbsoluteError'].max():,.2f}"
    )

    print(
        f"Mean APE : "
        f"{diagnostics['APE'].mean():.2f}%"
    )

    print(
        f"Median APE : "
        f"{diagnostics['APE'].median():.2f}%"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()