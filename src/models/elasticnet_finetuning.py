import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet

from src.models.baseline import (
    load_data,
    engineer_features,
    split_data,
    fit_preprocessor,
    transform_data,
    evaluate_model,
)


ALPHAS = [ 0.00001, 0.00003, 0.00005, 0.0001, 0.0002, 0.0003,]

L1_RATIOS = [ 0.80, 0.85, 0.90, 0.95, 0.99,]


def main():

    print("=" * 80)
    print("AMES HOUSING - ELASTIC NET FINE-TUNING")
    print("=" * 80)

   
    print("\n[1/4] Loading and preparing data...")
    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    

    print("\n[2/4] Preprocessing...")
    preprocessor = fit_preprocessor(X_train)
    X_train_processed, X_test_processed = transform_data(
        preprocessor,
        X_train,
        X_test,
    )

   
    print("\n[3/4] Transforming target...")
    y_train_log = np.log1p(y_train)

    

    print("\n[4/4] Testing configurations...")
    results = []
    total = len(ALPHAS) * len(L1_RATIOS)
    counter = 0
    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:
            counter += 1
            print(
                f"[{counter}/{total}] "
                f"alpha={alpha}, "
                f"l1_ratio={l1_ratio}"
            )

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=100000,
                tol=1e-4,
                random_state=42,
            )

            try:
                model.fit(  X_train_processed,    y_train_log,)
                predictions_log = model.predict(  X_test_processed )
                predictions = np.expm1( predictions_log )
                metrics = evaluate_model( y_test,  predictions,)

                results.append(
                    {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "R2": metrics["R2"],
                    }
                )

            except Exception as error:
                print( f"Failed: {error}")

    # ========================================================================
    # Results
    # ========================================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE"
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("TOP ELASTIC NET FINE-TUNING RESULTS")
    print("=" * 80)

    print(
        results_df.head(10).to_string(
            index=False
        )
    )

    # ========================================================================
    # Best configuration
    # ========================================================================

    best = results_df.iloc[0]

    print("\n" + "=" * 80)
    print("BEST ELASTIC NET CONFIGURATION")
    print("=" * 80)

    print(f"\nAlpha    : {best['alpha']}")
    print(f"L1 Ratio : {best['l1_ratio']}")
    print(f"MAE      : ${best['MAE']:,.2f}")
    print(f"RMSE     : ${best['RMSE']:,.2f}")
    print(f"R²       : {best['R2']:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()