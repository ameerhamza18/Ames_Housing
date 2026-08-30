from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from src.models.baseline import (
    load_data,
    engineer_features,
    split_data,
    fit_preprocessor,
    transform_data,
    evaluate_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir( parents=True, exist_ok=True,)



def main():

    print("=" * 80)
    print("AMES HOUSING - LOG TARGET LINEAR REGRESSION DIAGNOSTICS")
    print("=" * 80)

    print("\n[1/6] Loading dataset...")
    df = load_data()


    print("\n[2/6] Applying feature engineering...")
    df = engineer_features(df)

 

    print("\n[3/6] Creating train-test split...")
    X_train, X_test, y_train, y_test = split_data(df)


    print("\n[4/6] Preprocessing features...")
    preprocessor = fit_preprocessor(X_train)
    X_train_processed, X_test_processed = transform_data(
        preprocessor,
        X_train,
        X_test,
    )



    print("\n[5/6] Training log-target Linear Regression...")
    y_train_log = np.log1p(y_train)
    model = LinearRegression()
    model.fit( X_train_processed, y_train_log,)

    # Predict in log-space
    predictions_log = model.predict( X_test_processed )

    # Convert back to original dollar scale
    predictions = np.expm1( predictions_log)

    metrics = evaluate_model(y_test,predictions, )

    print("\n" + "=" * 80)
    print("MODEL RESULTS")
    print("=" * 80)

    print(f"\nMAE  : ${metrics['MAE']:,.2f}")
    print(f"MSE  : {metrics['MSE']:,.2f}")
    print(f"RMSE : ${metrics['RMSE']:,.2f}")
    print(f"R²   : {metrics['R2']:.4f}")



    print("\n[6/6] Performing residual diagnostics...")
    diagnostics = pd.DataFrame(
        {
            "Id": X_test["Id"].values,
            "ActualPrice": y_test.values,
            "PredictedPrice": predictions,
        }
    )

    diagnostics["Residual"] = ( diagnostics["ActualPrice"]  - diagnostics["PredictedPrice"])
    diagnostics["AbsoluteError"] = ( diagnostics["Residual"].abs() )
    diagnostics["AbsolutePercentageError"] = ( diagnostics["AbsoluteError"]  / diagnostics["ActualPrice"] * 100)

  
    print("\n" + "-" * 80)
    print("RESIDUAL STATISTICS")
    print("-" * 80)

    print( f"Mean residual   : " f"${diagnostics['Residual'].mean():,.2f}")
    print( f"Median residual : " f"${diagnostics['Residual'].median():,.2f}" )
    print( f"Max error       : " f"${diagnostics['AbsoluteError'].max():,.2f}" )
    print( f"Median abs error: " f"${diagnostics['AbsoluteError'].median():,.2f}" )
    print( f"Underpredictions: " f"{(diagnostics['Residual'] > 0).sum()}")
    print( f"Overpredictions : " f"{(diagnostics['Residual'] < 0).sum()}")


    print("\n" + "-" * 80)
    print("10 LARGEST PREDICTION ERRORS")
    print("-" * 80)

    largest_errors = ( diagnostics .sort_values( "AbsoluteError", ascending=False,).head(10))

    print(largest_errors[
            [ "Id", "ActualPrice", "PredictedPrice", "Residual", "AbsoluteError",]
        ].to_string(index=False)
    )

    # ========================================================================
    # Plot 1 — Actual vs Predicted
    # ========================================================================

    plt.figure(figsize=(8, 6))
    plt.scatter( diagnostics["ActualPrice"], diagnostics["PredictedPrice"], alpha=0.6,)

    minimum = min( diagnostics["ActualPrice"].min(),  diagnostics["PredictedPrice"].min(), )
    maximum = max( diagnostics["ActualPrice"].max(), diagnostics["PredictedPrice"].max(), )

    plt.plot( [minimum, maximum], [minimum, maximum], linestyle="--",)

    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png",dpi=300,)
    plt.close()

    # ========================================================================
    # Plot 2 — Residuals vs Predicted
    # ========================================================================

    plt.figure(figsize=(8, 6))
    plt.scatter( diagnostics["PredictedPrice"], diagnostics["Residual"], alpha=0.6, )

    plt.axhline( y=0, linestyle="--",)

    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Price")
    plt.tight_layout()
    plt.savefig( FIGURES_DIR / "residuals_vs_predicted.png", dpi=300,)
    plt.close()

    # ========================================================================
    # Plot 3 — Residual Distribution
    # ========================================================================

    plt.figure(figsize=(8, 6))
    plt.hist( diagnostics["Residual"], bins=30, )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig( FIGURES_DIR / "residual_distribution.png", dpi=300,)
    plt.close()

    # ========================================================================
    # Plot 4 — Actual Price vs Absolute Error
    # ========================================================================

    plt.figure(figsize=(8, 6))
    plt.scatter(  diagnostics["ActualPrice"],  diagnostics["AbsoluteError"],  alpha=0.6, )

    plt.xlabel("Actual SalePrice")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs Actual Price")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "absolute_error_vs_actual_price.png",dpi=300,)
    plt.close()

    print("\n" + "=" * 80)
    print("RESIDUAL DIAGNOSTICS COMPLETED")
    print("\nDiagnostic plots saved:")

    print(f"  - {FIGURES_DIR / 'actual_vs_predicted.png'}")
    print(f"  - {FIGURES_DIR / 'residuals_vs_predicted.png'}")
    print(f"  - {FIGURES_DIR / 'residual_distribution.png'}")
    print(f"  - {FIGURES_DIR / 'absolute_error_vs_actual_price.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()