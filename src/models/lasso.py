import numpy as np

from sklearn.linear_model import Lasso

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
    print("AMES HOUSING - LASSO REGRESSION WITH LOG TARGET")
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


    print("\n[5/5] Training Lasso...")
    y_train_log = np.log1p(y_train)
    model = Lasso(
        alpha=0.0002,
        max_iter=100000,
        tol=1e-4,
    )

    model.fit(  X_train_processed,  y_train_log,)
    predictions_log = model.predict(X_test_processed, )
    predictions = np.expm1( predictions_log,)
    metrics = evaluate_model( y_test, predictions,)

    

    print("\n" + "=" * 80)
    print("LASSO RESULTS")
    print("=" * 80)

    print( f"\nAlpha : {model.alpha}")
    print( f"MAE   : ${metrics['MAE']:,.2f}")
    print( f"RMSE  : ${metrics['RMSE']:,.2f}")
    print( f"R²    : {metrics['R2']:.4f}")

    # Number of non-zero coefficients
    non_zero = np.count_nonzero( model.coef_)
    total_features = len( model.coef_)

    print(  f"\nNon-zero coefficients: "
        f"{non_zero}/{total_features}"
    )

    print(
        f"Sparsity: "
        f"{(1 - non_zero / total_features) * 100:.2f}%"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()