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

OUTLIER_IDS = [1299, 524]

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


def evaluate_dataset(df, experiment_name):

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = np.log1p(
        df[TARGET_COLUMN]
    )

    numerical_features, categorical_features = (
        identify_features(df)
    )

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
    )

    mae = -results["test_MAE"]
    rmse = -results["test_RMSE"]
    r2 = results["test_R2"]

    print("\n" + "=" * 80)
    print(experiment_name)
    print("=" * 80)

    print(
        f"\nMean MAE  : {mae.mean():.4f}"
    )

    print(
        f"Std MAE   : {mae.std():.4f}"
    )

    print(
        f"\nMean RMSE : {rmse.mean():.4f}"
    )

    print(
        f"Std RMSE  : {rmse.std():.4f}"
    )

    print(
        f"\nMean R²   : {r2.mean():.4f}"
    )

    print(
        f"Std R²    : {r2.std():.4f}"
    )

    print("\nFold results:")

    fold_results = pd.DataFrame(
        {
            "Fold": range(1, N_SPLITS + 1),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    print(
        fold_results.to_string(
            index=False
        )
    )


def main():

    print("=" * 80)
    print("AMES HOUSING - OUTLIER ABLATION STUDY")
    print("=" * 80)

    # ========================================================================
    # Load + feature engineering
    # ========================================================================

    df = load_data()

    df = engineer_features(df)

    print(
        f"\nOriginal samples: {len(df)}"
    )

    print(
        f"Candidate outliers: {OUTLIER_IDS}"
    )

    # ========================================================================
    # Experiment A
    # ========================================================================

    evaluate_dataset(
        df,
        "EXPERIMENT A - ORIGINAL DATASET",
    )

    # ========================================================================
    # Experiment B
    # ========================================================================

    filtered_df = df[
        ~df[ID_COLUMN].isin(OUTLIER_IDS)
    ].copy()

    print(
        f"\nSamples after removing candidates: "
        f"{len(filtered_df)}"
    )

    evaluate_dataset(
        filtered_df,
        "EXPERIMENT B - WITHOUT CANDIDATE OUTLIERS",
    )

    print("\n" + "=" * 80)
    print("OUTLIER ABLATION STUDY COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()