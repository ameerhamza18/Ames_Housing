from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

# Defining the project path (...Hamza)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data"/ "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "ames_housing.csv"

# Now Dataset configuration
OPENML_DATA_ID = 42165
TARGET_COLUMN ="SalePrice"





# Download dataset
def download_dataset() -> pd.DataFrame:
    print(f"Downloading the dataset from OpenML with ID {OPENML_DATA_ID}...")
    dataset = fetch_openml(
        data_id=OPENML_DATA_ID,
        as_frame=True,
        parser = "auto"
    )

    X = dataset.data.copy()
    y = dataset.target.copy()

    if y.name != TARGET_COLUMN:
        y.name = TARGET_COLUMN

    df = pd.concat([X, y], axis=1)
    print(f"Dataset downloaded successfully with shape: {df.shape}")
    return df


     


# validate the dataset
def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The dataset is empty. Please check the download process.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"The target column '{TARGET_COLUMN}' is missing from the dataset.")
    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(f"The target column '{TARGET_COLUMN}' contains missing values.")

    print("Basic validation passed: Dataset is not empty, target column exists, and no missing values in the target column.")





# Save raw dataset
def save_raw_dataset(df:pd.DataFrame) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw dataset saved successfully at {RAW_DATA_PATH}")




# Main Execution
def main() -> None:
    df = download_dataset()
    validate_dataset(df)
    save_raw_dataset(df)

    print("\n Dataset download and validation completed successfully.")
    print(f"Rows:{df.shape[0]}")
    print(f"Columns:{df.shape[1]}")

if __name__ == "__main__":
    main()