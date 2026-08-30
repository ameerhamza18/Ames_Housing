import numpy as np

from sklearn.linear_model import ElasticNet

from src.models.baseline import (
    load_data,
    engineer_features,
    split_data,
    fit_preprocessor,
    transform_data,
    evaluate_model,
)


def main():

    print("=" * 80)
    print("AMES HOUSING - ELASTIC NET WITH LOG TARGET")
    print("=" * 80)

 
    print("\n[1/5] Loading dataset...")
    df = load_data()

   
    print("\n[2/5] Applying feature engineering...")
    df = engineer_features(df)


    print("\n[3/5] Creating train-test split...")
    X_train, X_test, y_train, y_test = split_data(df)


    print("\n[4/5] Preprocessing features...")
    preprocessor = fit_preprocessor(X_train)
    X_train_processed, X_test_processed = transform_data(
        preprocessor,
        X_train,
        X_test,
    )

    # ========================================================================
    #  ElasticNet
    # ========================================================================

    print("\n[5/5] Training ElasticNet...")
    # Same target transformation as our current champion
    y_train_log = np.log1p(y_train)

    model = ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=42,
    )

    model.fit(
        X_train_processed,
        y_train_log,
    )

    # Predict in log-space
    predictions_log = model.predict(
        X_test_processed,
    )

    # Convert back to dollar scale
    predictions = np.expm1(
        predictions_log,
    )

    # Evaluate on original SalePrice
    metrics = evaluate_model(
        y_test,
        predictions,
    )

    # ========================================================================
    # Results
    # ========================================================================

    print("\n" + "=" * 80)
    print("ELASTIC NET RESULTS")
    print("=" * 80)

    print(
        f"\nAlpha    : {model.alpha}"
    )

    print(
        f"L1 Ratio : {model.l1_ratio}"
    )

    print(
        f"MAE      : ${metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE     : ${metrics['RMSE']:,.2f}"
    )

    print(
        f"R²       : {metrics['R2']:.4f}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()