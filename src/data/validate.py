from pathlib import Path
import json
import pandas as pd

#Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data"/ "raw" / "ames_housing.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROFILE_REPORT_PATH = REPORTS_DIR / "data_profile.json"



# Dataset configuartion
EXCEPTED_COLUMN_COUNT =81
ID_COLUMN ="Id"
TARGET_COLUMN = "SalePrice"



#Load dataset
def load_raw_dataset() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset loaded successfully: {df.shape}")
    return df


# Validation Schema
def validate_schema(df:pd.DataFrame) -> None:
    if df.empty:
        raise ValueError ("Dataset is empty")
    if len(df.columns) != EXCEPTED_COLUMN_COUNT:
        raise ValueError(f"Expected {EXCEPTED_COLUMN_COUNT} columns,"
                         f" but found {len(df.columns)} columns")
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Missing expected ID column: {ID_COLUMN}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing expected target column: {TARGET_COLUMN}")
    print("Schema validation passed successfully.")



# Data quality validation
def validate_data_quality(df: pd.DataFrame) -> dict:
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
    missing_target = int(df[TARGET_COLUMN].isna().sum())

    constant_columns =[
        column for column in df.columns if df[column].nunique(dropna=False) <=1
    ]

    quality_report = {
        "row_count": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "missing_target": missing_target,
        "constant_columns": constant_columns
    }
    print("\nData Quality Report:")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Missing target values: {missing_target}")
    print(f"Constant columns: {constant_columns}")

    if missing_target > 0:
        raise ValueError(f"Missing values found in target column '{TARGET_COLUMN}'",
                         f"Count: {missing_target}")
    print("Data quality validation passed successfully.")
    return quality_report



# Missing-values profiling
def profile_missing_values(df:pd.DataFrame) -> list:
    missing = df.isna().sum()
    missing_report = []
    for column in df.columns:
        missing_count = int(missing[column])
        missing_report.append({
            "column": column,
            "missing_count": missing_count,
            "missing_percentage":round((missing_count / len(df)) * 100, 2)
        })
    return missing_report



# Feature Profiling
def profile_feature(df: pd.DataFrame) -> list:
    feature_report = []
    for column in df.columns:
        series = df[column]
        profile={
            "column": column,
            "data_type": str(series.dtype),
            "unique_values": int(series.nunique(dropna=False)),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round((series.isna().sum() / len(df)) * 100, 2)
        }
        if pd.api.types.is_numeric_dtype(series):
            profile.update({
                "min":float(series.min())
                if not series.dropna().empty else None,
                "max": float(series.max())
                if not series.dropna().empty else None,
                "mean": float(series.mean())
                if not series.dropna().empty else None,
                "std": float(series.std())
                if not series.dropna().empty else None
            })
        else:
            profile.update({
                "min":None,
                "max": None,
                "mean": None,
                "std": None
            })

        feature_report.append(profile)
    return feature_report




# Generate complete profile report

def generate_profile_report(df:pd.DataFrame, quality_report:dict)->dict:
    report={
        "dataset":{
            "name":"Ames Housing",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target": TARGET_COLUMN,
            "identifier": ID_COLUMN
        },
        "quality": quality_report,
        "missing_values":profile_missing_values(df),
        "features": profile_feature(df)
    }
    return report



# Save profile report to JSON file
def save_profile_report(report:dict)-> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(
        PROFILE_REPORT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file , indent=4)
    print(f"Profile report saved successfully at {PROFILE_REPORT_PATH}")




# Main execution
def main()->None:
    print("="* 80)
    print("AMES Housing --- Data Validation & Profiling")
    print("="* 80)

    df = load_raw_dataset()
    validate_schema(df)
    quality_report = validate_data_quality(df)
    report = generate_profile_report(df, quality_report)
    save_profile_report(report)

    print("\nValidation and profiling completed successfully.")



if __name__ =="__main__":
    main()
