import numpy as np

from sklearn.linear_model import Ridge

from src.models.baseline import (
    load_data,
    engineer_features,
    split_data,
    fit_preprocessor,
    transform_data,
    evaluate_model,
)


ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


def main():

    print("=" * 80)
    print("AMES HOUSING - RIDGE HYPERPARAMETER EXPERIMENT")
    print("=" * 80)

    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)



    preprocessor = fit_preprocessor(X_train)
    X_train_processed, X_test_processed = transform_data(
        preprocessor,
        X_train,
        X_test,
    )

  
    y_train_log = np.log1p(y_train)
    results = []
    for alpha in ALPHAS:
        print(f"\nTesting alpha = {alpha}")
        model = Ridge(alpha=alpha )
        model.fit(
            X_train_processed,
            y_train_log,
        )

        predictions_log = model.predict(  X_test_processed)
        predictions = np.expm1(predictions_log)
        metrics = evaluate_model( y_test, predictions, )
        results.append(
            {
                "alpha": alpha,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    # ------------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RIDGE HYPERPARAMETER RESULTS")
    print("=" * 80)

    print(
        f"{'Alpha':>10}"
        f"{'MAE':>15}"
        f"{'RMSE':>15}"
        f"{'R²':>12}"
    )

    print("-" * 55)

    for result in results:

        print(
            f"{result['alpha']:>10}"
            f"${result['MAE']:>14,.2f}"
            f"${result['RMSE']:>14,.2f}"
            f"{result['R2']:>12.4f}"
        )

    # ------------------------------------------------------------------------
    # Best Ridge model
    # ------------------------------------------------------------------------

    best = min(
        results,
        key=lambda x: x["RMSE"]
    )

    print("\n" + "=" * 80)
    print("BEST RIDGE CONFIGURATION")
    print("=" * 80)

    print( f"\nAlpha : {best['alpha']}")
    print( f"MAE   : ${best['MAE']:,.2f}" )
    print( f"RMSE  : ${best['RMSE']:,.2f}")
    print( f"R²    : {best['R2']:.4f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()