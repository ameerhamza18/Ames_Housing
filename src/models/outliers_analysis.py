import pandas as pd
import numpy as np

from src.models.baseline import load_data
from src.features.engineer import engineer_features


TARGET_COLUMN = "SalePrice"


def main():

    print("=" * 80)
    print("AMES HOUSING - OUTLIER ANALYSIS")
    print("=" * 80)

    # ========================================================================
    # Load data
    # ========================================================================

    print("\n[1/5] Loading dataset...")

    df = load_data()

    df = engineer_features(df)

    # ========================================================================
    # Target statistics
    # ========================================================================

    print("\n[2/5] SalePrice statistics...")

    print(
        df[TARGET_COLUMN].describe()
    )

    # ========================================================================
    # IQR detection
    # ========================================================================

    print("\n[3/5] Detecting SalePrice outliers...")

    q1 = df[TARGET_COLUMN].quantile(0.25)
    q3 = df[TARGET_COLUMN].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    target_outliers = df[
        (df[TARGET_COLUMN] < lower_bound)
        | (df[TARGET_COLUMN] > upper_bound)
    ]

    print(
        f"\nQ1: ${q1:,.2f}"
    )

    print(
        f"Q3: ${q3:,.2f}"
    )

    print(
        f"IQR: ${iqr:,.2f}"
    )

    print(
        f"Lower bound: ${lower_bound:,.2f}"
    )

    print(
        f"Upper bound: ${upper_bound:,.2f}"
    )

    print(
        f"\nTarget outliers detected: "
        f"{len(target_outliers)}"
    )

    # ========================================================================
    # Extreme living area
    # ========================================================================

    print("\n[4/5] Detecting extreme living areas...")

    area_threshold = (
        df["GrLivArea"].quantile(0.99)
    )

    large_houses = df[
        df["GrLivArea"] > area_threshold
    ].copy()

    print(
        f"\n99th percentile GrLivArea: "
        f"{area_threshold:,.2f}"
    )

    print(
        f"Houses above threshold: "
        f"{len(large_houses)}"
    )

    # ========================================================================
    # Suspicious combinations
    # ========================================================================

    print("\n[5/5] Detecting suspicious price/area combinations...")

    suspicious = df[
        (
            (df["GrLivArea"] > 4000)
            & (df[TARGET_COLUMN] < 300000)
        )
        |
        (
            (df["TotalBsmtSF"] > 4000)
            & (df[TARGET_COLUMN] < 300000)
        )
    ].copy()

    print("\n" + "=" * 80)
    print("SUSPICIOUS OBSERVATIONS")
    print("=" * 80)

    columns = [
        "Id",
        TARGET_COLUMN,
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "TotalHouseSF",
        "Neighborhood",
        "YearBuilt",
    ]

    print(
        suspicious[columns]
        .sort_values(
            TARGET_COLUMN
        )
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("OUTLIER ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()