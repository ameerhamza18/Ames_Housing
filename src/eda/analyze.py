from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import pandas as pd



# Project path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ames_housing.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_REPORT_PATH = REPORTS_DIR / "eda_summary.txt"


# Configuration
TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"



# Load dataset
def load_dataset() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at:{RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print("Dataset loaded successfully")
    return df



# Dataset Overview
def analyze_dataset_structure(df: pd.DataFrame)-> dict:
    numerical_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    if ID_COLUMN in numerical_columns:
        numerical_columns.remove(ID_COLUMN)
    summary={
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_features":len(numerical_columns),
        "categorical_features": len(categorical_columns)
    }
    return summary



# Missing Values analysis
def analyze_missing_values(df : pd.DataFrame) -> pd.DataFrame:
    missing= (
        df.isna().sum().to_frame(name="missing_count")
    )

    missing["missing_percenatge"] =(
        missing["missing_count"] / len(df) *100
    )

    missing = (
        missing[missing["missing_count"] > 0].sort_values(
            "missing_percenatge", ascending=False
        )
    )
    return missing



# Target Analysis
def analyze_target(df: pd.DataFrame)-> dict:
    target = df[TARGET_COLUMN]

    statistics ={
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        # "skewness": float(target.skew())
    }
    return statistics


# Numerical Analysis
def analyze_correlations(df:pd.DataFrame) -> pd.DataFrame:
    numerical_df = df.select_dtypes(include="number")
    correlations =(
        numerical_df.corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN).sort_values(
            key=lambda series: series.abs(), ascending=False
        )
    )
    return correlations.to_frame(
        name="correlation_with_saleprice"
    )



# Outliers analysis
def analyze_outliers(df:pd.DataFrame) -> pd.DataFrame:
    numerical_columns = df.select_dtypes(include="number")
    results =[]

    for column in numerical_columns:
        if column == ID_COLUMN:
            continue
        series = df[column].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 -q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = (
            (series < lower_bound) | (series > upper_bound)
        ).sum()

        results.append({
            "column": column,
            "outlier_count": outlier_count,
            "outlier_percentage":round(outlier_count / len(series)
                                       * 100 , 2,)
        })

    return (
            pd.DataFrame(results).sort_values(
                "outlier_percentage", ascending=False
            )
        )




# Plot target distribution
def plot_target_distribution(df: pd.DataFrame) -> None:
        FIGURES_DIR.mkdir(parents=True,exist_ok=True,)

        plt.figure(figsize=(10, 6))
        sns.histplot( data=df, x=TARGET_COLUMN,kde=True,)

        plt.title("SalePrice Distribution")
        plt.xlabel("Sale Price")
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
        FIGURES_DIR / "target_distribution.png",
        dpi=150,)
        plt.close()



# Log Target distribution
def plot_log_target_distribution(df: pd.DataFrame) -> None:
    log_target = np.log1p(df[TARGET_COLUMN])

    plt.figure(figsize=(10, 6))
    sns.histplot( log_target, kde=True,)

    plt.title("Log-Transformed SalePrice Distribution")
    plt.xlabel("log1p(SalePrice)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig( FIGURES_DIR / "log_target_distribution.png",
                 dpi=150,
    )
    plt.close()




# Plot: correlation heatmap
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numerical_df = df.select_dtypes(include="number" )

    correlations = ( numerical_df .corr()[TARGET_COLUMN] .abs()
        .sort_values(  ascending=False, ))
    
    top_features = correlations.head(11).index
    correlation_matrix = ( numerical_df[top_features] .corr() )

    plt.figure(figsize=(11, 8))
    sns.heatmap(correlation_matrix,annot=True,fmt=".2f",
        cmap="coolwarm",center=0,)

    plt.title( "Correlation Matrix — Strongest Numerical Features")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "correlation_heatmap.png",
        dpi=150,
    )
    plt.close()






# Generate EDA report
def generate_report(structure: dict,missing_values: pd.DataFrame,
    target_statistics: dict,correlations: pd.DataFrame,
    outliers: pd.DataFrame,) -> None:

    REPORTS_DIR.mkdir(parents=True,exist_ok=True, )

    with open( EDA_REPORT_PATH, "w", encoding="utf-8", ) as file:

        file.write("AMES HOUSING — EDA SUMMARY\n")
        file.write("=" * 70)
        file.write("\n\n")

        file.write("DATASET STRUCTURE\n")
        file.write("-" * 70)
        file.write("\n")

        file.write(f"Rows: {structure['rows']}\n" )
        file.write(f"Columns: {structure['columns']}\n")

        file.write( f"Numerical features: " f"{structure['numerical_features']}\n")
        file.write( f"Categorical features: "f"{structure['categorical_features']}\n\n")

        file.write("TARGET ANALYSIS\n")
        file.write("-" * 70)
        file.write("\n")

        for key, value in target_statistics.items():
            file.write( f"{key}: {value:,.4f}\n")
            file.write("\n")

        file.write("MISSING VALUES\n")
        file.write("-" * 70)
        file.write("\n")

        if missing_values.empty:
            file.write( "No missing values found.\n")
        else:
            file.write(  missing_values.to_string())
            file.write("\n")
        file.write("\n")

        file.write("STRONGEST CORRELATIONS WITH SalePrice\n")
        file.write("-" * 70)
        file.write("\n")
        file.write( correlations.head(15).to_string() )
        file.write("\n\n")

        file.write("NUMERICAL OUTLIER SUMMARY\n")
        file.write("-" * 70)
        file.write("\n")
        file.write(outliers.head(15).to_string(index=False) )

        file.write("\n")



# Main Execution
def main()-> None:
    print("=" * 80)
    print("Ames Housing --- Exploratory Data Analysis")
    print("=" * 80)

    df = load_dataset()

    print("\nAnalyzing dataset structure...")
    structure = analyze_dataset_structure(df)
    print( f"Numerical features: " f"{structure['numerical_features']}")
    print( f"Categorical features: " f"{structure['categorical_features']}" )

    print("\nAnalyzing missing values...")
    missing_values = analyze_missing_values(df)
    print( f"Columns containing missing values: " f"{len(missing_values)}")

    print("\nAnalyzing target variable...")
    target_statistics = analyze_target(df)
    print(f"Mean SalePrice: "f"${target_statistics['mean']:,.2f}")
    print(f"Median SalePrice: "f"${target_statistics['median']:,.2f}")

    print("\nAnalyzing correlations...")
    correlations = analyze_correlations(df)
    print(correlations.head(10))

    print("\nAnalyzing numerical outliers...")
    outliers = analyze_outliers(df)
    print( outliers.head(10))

    print("\nGenerating visualizations...")
    plot_target_distribution(df)
    plot_log_target_distribution(df)
    plot_correlation_heatmap(df)

    print("\nGenerating EDA report...")
    generate_report( structure, missing_values, target_statistics,
        correlations, outliers,)
    print("\nEDA completed successfully.")

    print( f"Report: {EDA_REPORT_PATH}")
    print( f"Figures: {FIGURES_DIR}" )



if __name__ == "__main__":
    main()

