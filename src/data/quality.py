from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"


# These observations were identified during the
# cross-validation outlier investigation.
#
# Reason:
# Extremely unusual price/size combinations that
# disproportionately destabilize model validation.
#
# IMPORTANT:
# Raw data is never modified.

KNOWN_INFLUENTIAL_IDS = [
    1299,
    524,
]


def remove_influential_observations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove observations identified as highly influential
    during model diagnostics.

    The original dataframe is not modified.
    """

    df = df.copy()

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{ID_COLUMN}' not found."
        )

    before = len(df)

    df = df[
        ~df[ID_COLUMN].isin(
            KNOWN_INFLUENTIAL_IDS
        )
    ].copy()

    after = len(df)

    removed = before - after

    print(
        f"Influential observations removed: {removed}"
    )

    return df


def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Perform basic dataset integrity checks.
    """

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "is missing."
        )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"ID column '{ID_COLUMN}' "
            "is missing."
        )

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate IDs detected."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Missing target values detected."
        )

    if (df[TARGET_COLUMN] <= 0).any():
        raise ValueError(
            "SalePrice must contain only "
            "positive values."
        )