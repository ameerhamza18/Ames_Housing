from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ( mean_absolute_error,mean_squared_error,r2_score,)
from src.features.engineer import (engineer_features,)
from src.features.preprocess import (load_data,split_data,fit_preprocessor,transform_data,)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_REPORT_PATH = (REPORTS_DIR / "baseline_results.txt")
PREDICTIONS_PATH = (
    REPORTS_DIR / "baseline_predictions.csv"
)
RANDOM_STATE = 42

# EVALUATION
# ============================================================================
def evaluate_model(y_true:pd.Series,predictions:np.ndarray,) -> dict:
    mae = mean_absolute_error( y_true, predictions,)
    mse = mean_squared_error(y_true,predictions,)
    rmse = np.sqrt(mse)
    r2 = r2_score( y_true, predictions,)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


# BUILD PREDICTION DATAFRAME
# ============================================================================

def build_prediction_dataframe( X_test: pd.DataFrame, y_test: pd.Series,
                               predictions: np.ndarray,) -> pd.DataFrame:
        results = pd.DataFrame(
        {
            "Id": X_test["Id"].values,
            "ActualPrice": y_test.values,
            "PredictedPrice": predictions,
        })

        results["Residual"] = (
        results["ActualPrice"]- results["PredictedPrice"])

        results["AbsoluteError"] = ( results["Residual"].abs())
        results["AbsolutePercentageError"] = (results["AbsoluteError"]
        / results["ActualPrice"] * 100)

        return results




# ERROR ANALYSIS
# ============================================================================

def analyze_errors(prediction_df: pd.DataFrame,) -> dict:
    
    largest_errors = ( prediction_df .sort_values(
            "AbsoluteError", ascending=False,).head(10))

    
    mean_residual = ( prediction_df["Residual"].mean())

    median_absolute_error = (prediction_df["AbsoluteError"].median())

    maximum_absolute_error = (prediction_df["AbsoluteError"].max())

    underpredictions = ( prediction_df["Residual"] > 0 ).sum()

    overpredictions = ( prediction_df["Residual"] < 0).sum()

    return {
        "mean_residual": mean_residual,
        "median_absolute_error": median_absolute_error,
        "maximum_absolute_error": maximum_absolute_error,
        "underpredictions": underpredictions,
        "overpredictions": overpredictions,
        "largest_errors": largest_errors,
    }



# PRICE-BAND ERROR ANALYSIS
# ============================================================================

def analyze_price_bands(prediction_df: pd.DataFrame,) -> pd.DataFrame:

    bands = pd.cut( prediction_df["ActualPrice"],
        bins=[ 0, 100_000, 200_000,
            300_000, 500_000, np.inf,],

        labels=["<=100K", "100K-200K", "200K-300K",
            "300K-500K", ">500K", ],
        include_lowest=True,
    )

    analysis = ( prediction_df .assign(PriceBand=bands)
        .groupby( "PriceBand", observed=False,
        )
        .agg(
            Samples=("ActualPrice", "count"),
            MAE=("AbsoluteError", "mean"),
            RMSE=("Residual",
                lambda x: np.sqrt(np.mean(x ** 2)),
            ),
            MeanActualPrice=( "ActualPrice", "mean",),
        )
        .reset_index()
    )

    return analysis






# SAVE RESULTS
# ============================================================================


def save_report(metrics: dict,error_analysis: dict,price_band_analysis: pd.DataFrame,) -> None:

    REPORTS_DIR.mkdir( parents=True, exist_ok=True,)

    with open( BASELINE_REPORT_PATH, "w", encoding="utf-8",) as file:
        file.write( "AMES HOUSING - BASELINE LINEAR REGRESSION\n")
        file.write("=" * 80)
        file.write("\n\n")

  
        file.write("MODEL\n")
        file.write("-" * 80)
        file.write("\n")
        file.write( "Linear Regression\n" )
        file.write(f"Random State: {RANDOM_STATE}\n\n")


        file.write( "EVALUATION METRICS\n")
        file.write("-" * 80)
        file.write("\n")
        file.write( f"MAE  : ${metrics['MAE']:,.2f}\n")
        file.write( f"MSE  : {metrics['MSE']:,.2f}\n")
        file.write( f"RMSE : ${metrics['RMSE']:,.2f}\n")
        file.write( f"R2   : {metrics['R2']:.4f}\n\n")

   
        file.write( "ERROR ANALYSIS\n")
        file.write("-" * 80)
        file.write("\n")
        file.write(
            f"Mean Residual          : "
            f"${error_analysis['mean_residual']:,.2f}\n"
            )
        file.write(
            f"Median Absolute Error  : "
            f"${error_analysis['median_absolute_error']:,.2f}\n"
        )

        file.write(
            f"Maximum Absolute Error : "
            f"${error_analysis['maximum_absolute_error']:,.2f}\n"
        )

        file.write(
            f"Underpredictions       : "
            f"{error_analysis['underpredictions']}\n"
        )

        file.write(
            f"Overpredictions        : "
            f"{error_analysis['overpredictions']}\n\n"
        )

       
        file.write( "ERROR BY ACTUAL PRICE BAND\n")
        file.write("-" * 80)
        file.write("\n\n")
        file.write( price_band_analysis .to_string(index=False) )
        file.write("\n\n")

      
        file.write("10 LARGEST PREDICTION ERRORS\n")
        file.write("-" * 80)
        file.write("\n\n")
        file.write(error_analysis[  "largest_errors"]
            .to_string(index=False)
        )




# SAVE PREDICTIONS
# ============================================================================

def save_predictions( prediction_df: pd.DataFrame,) -> None:
  
    REPORTS_DIR.mkdir(parents=True,exist_ok=True,)

    prediction_df.to_csv( PREDICTIONS_PATH,index=False, )



# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print("AMES HOUSING - BASELINE LINEAR REGRESSION")
    print("=" * 80)

    
    # 1. Load dataset...................
    print( "\n[1/9] Loading dataset...")
    df = load_data()
    print( f"Original dataset shape: {df.shape}")

    
    # 2. Feature engineering.................   
    print("\n[2/9] Applying existing feature engineering...")
    df = engineer_features(df)
    print(f"Engineered dataset shape: {df.shape}")


    # 3. Train-test split....................
    print( "\n[3/9] Creating train-test split...")
    ( X_train,  X_test,  y_train,  y_test,) = split_data(df)
    print(f"Training samples: {len(X_train)}")
    print(  f"Testing samples: {len(X_test)}")


    # 4. Fit preprocessor................. 
    print( "\n[4/9] Fitting existing preprocessing pipeline...")
    preprocessor = fit_preprocessor( X_train=X_train)
    print( "Preprocessor fitted successfully.")

    
    # 5. Transform data
    print( "\n[5/9] Transforming train and test features..." )
    ( X_train_processed, X_test_processed,) = transform_data( preprocessor,  X_train,X_test,)

    print( f"Processed training shape: "  f"{X_train_processed.shape}")
    print(
        f"Processed testing shape: "
        f"{X_test_processed.shape}"
    )


    # 6. Train model...........................
    print( "\n[6/9] Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train_processed,y_train,)
    print("Linear Regression trained successfully.")


    # 7. Predictions + metrics.........................
    print( "\n[7/9] Generating predictions and evaluating...")
    predictions = model.predict(  X_test_processed)
    metrics = evaluate_model(y_test,predictions,)

    print("\n" + "=" * 80)
    print("BASELINE MODEL RESULTS")
    print("=" * 80)

    print(f"\nMAE  : ${metrics['MAE']:,.2f}")
    print(f"MSE  : {metrics['MSE']:,.2f}" )
    print( f"RMSE : ${metrics['RMSE']:,.2f}")
    print(  f"R²   : {metrics['R2']:.4f}")

  
    # 8. Error analysis.......

    print( "\n[8/9] Performing error analysis...")
    prediction_df = build_prediction_dataframe(  X_test, y_test, predictions,)
    error_analysis = analyze_errors( prediction_df)

    price_band_analysis = analyze_price_bands(prediction_df)
    print(f"Mean residual: "f"${error_analysis['mean_residual']:,.2f}")

    print(f"Median absolute error: "f"${error_analysis['median_absolute_error']:,.2f}")
    print( f"Maximum absolute error: " f"${error_analysis['maximum_absolute_error']:,.2f}")

    print( f"Underpredictions: " f"{error_analysis['underpredictions']}")
    print(f"Overpredictions: "f"{error_analysis['overpredictions']}")

  
    # 9. Save experiment artifacts
    print( "\n[9/9] Saving experiment artifacts...")

    save_report( metrics, error_analysis, price_band_analysis, )
    save_predictions( prediction_df)

    print(  f"\nReport saved to:\n"  f"{BASELINE_REPORT_PATH}")
    print( f"\nPredictions saved to:\n" f"{PREDICTIONS_PATH}")

    print("\n" + "=" * 80)
    print( "BASELINE MODEL + ERROR ANALYSIS COMPLETED" )
    print("=" * 80)


if __name__ == "__main__":
    main()