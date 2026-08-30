import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

from src.models.baseline import (
    load_data,
    engineer_features,
    split_data,
    fit_preprocessor,
    transform_data,
    evaluate_model,
)


ALPHAS = [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
]

L1_RATIOS = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
]


def main():

    print("=" * 80)
    print("AMES HOUSING - ELASTIC NET HYPERPARAMETER SEARCH")
    print("=" * 80)

    # ========================================================================
    # 1. Load + feature engineering
    # ========================================================================

    print("\n[1/4] Loading and preparing data...")

    df = load_data()

    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # ========================================================================
    # 2. Preprocessing
    # ========================================================================

    print("\n[2/4] Preprocessing...")

    preprocessor = fit_preprocessor(X_train)

    X_train_processed, X_test_processed = transform_data(
        preprocessor,
        X_train,
        X_test,
    )

    # ========================================================================
    # 3. Log target
    # ========================================================================

    print("\n[3/4] Transforming target...")

    y_train_log = np.log1p(y_train)

    # ========================================================================
    # 4. Hyperparameter search
    # ========================================================================

    print("\n[4/4] Testing ElasticNet configurations...")

    results = []

    total_experiments = len(ALPHAS) * len(L1_RATIOS)
    experiment_number = 0

    for alpha in ALPHAS:

        for l1_ratio in L1_RATIOS:

            experiment_number += 1

            print(
                f"\n[{experiment_number}/{total_experiments}] "
                f"alpha={alpha}, l1_ratio={l1_ratio}"
            )

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=100000,
                tol=1e-4,
                random_state=42,
            )

            try:

                model.fit(
                    X_train_processed,
                    y_train_log,
                )

                predictions_log = model.predict(
                    X_test_processed
                )

                predictions = np.expm1(
                    predictions_log
                )

                metrics = evaluate_model(
                    y_test,
                    predictions,
                )

                results.append(
                    {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "R2": metrics["R2"],
                        "status": "success",
                    }
                )

            except Exception as error:

                print(
                    f"Configuration failed: {error}"
                )

                results.append(
                    {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "status": "failed",
                    }
                )

    # ========================================================================
    # Results
    # ========================================================================

    results_df = pd.DataFrame(results)

    successful_results = results_df[
        results_df["status"] == "success"
    ].copy()

    successful_results = successful_results.sort_values(
        by="RMSE"
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("ELASTIC NET RESULTS")
    print("=" * 80)

    print(
        successful_results.to_string(
            index=False
        )
    )

    # ========================================================================
    # Best model
    # ========================================================================

    if successful_results.empty:

        raise RuntimeError(
            "No ElasticNet configuration converged successfully."
        )

    best = successful_results.iloc[0]

    print("\n" + "=" * 80)
    print("BEST ELASTIC NET CONFIGURATION")
    print("=" * 80)

    print(
        f"\nAlpha    : {best['alpha']}"
    )

    print(
        f"L1 Ratio : {best['l1_ratio']}"
    )

    print(
        f"MAE      : ${best['MAE']:,.2f}"
    )

    print(
        f"RMSE     : ${best['RMSE']:,.2f}"
    )

    print(
        f"R²       : {best['R2']:.4f}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()