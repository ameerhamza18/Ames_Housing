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


def main():

    print("=" * 80)
    print("AMES HOUSING - RIDGE REGRESSION WITH LOG TARGET")
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


    print("\n[5/5] Training Ridge Regression...")
    # Same target transformation as our current best model
    y_train_log = np.log1p(y_train)
    model = Ridge( alpha=10.0 )

    model.fit(  X_train_processed,  y_train_log, )

    # Predict in log-space
    predictions_log = model.predict(  X_test_processed)

    # Convert back to original dollar scale
    predictions = np.expm1(predictions_log)

    # Evaluate on original SalePrice
    metrics = evaluate_model( y_test, predictions,)

    # ========================================================================
    # Results
    # ========================================================================

    print("\n" + "=" * 80)
    print("RIDGE REGRESSION RESULTS")
    print("=" * 80)

    print( f"\nAlpha: 10.0")
    print( f"MAE  : ${metrics['MAE']:,.2f}")
    print( f"MSE  : {metrics['MSE']:,.2f}")
    print( f"RMSE : ${metrics['RMSE']:,.2f}")
    print( f"R²   : {metrics['R2']:.4f}" )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()