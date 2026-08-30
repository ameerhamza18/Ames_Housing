import pandas as pd

from src.models.baseline import load_data
from src.features.engineer import engineer_features


TARGET_COLUMN = "SalePrice"

ERROR_IDS = [
    1299,
    524,
    826,
    1183,
    186,
    1325,
    689,
    804,
    899,
    692,
    1182,
    1047,
    775,
    582,
    633,
    589,
    584,
    474,
    319,
    886,
]


def main():

    print("=" * 80)
    print("AMES HOUSING - ERROR ANALYSIS")
    print("=" * 80)

    # ========================================================================
    # Load data
    # ========================================================================

    print("\n[1/3] Loading dataset...")

    df = load_data()

    # ========================================================================
    # Feature engineering
    # ========================================================================

    print("\n[2/3] Applying feature engineering...")

    df = engineer_features(df)

    # ========================================================================
    # Select problematic observations
    # ========================================================================

    print("\n[3/3] Inspecting high-error observations...")

    error_df = df[
        df["Id"].isin(ERROR_IDS)
    ].copy()

    # Important features for understanding price
    important_columns = [
        "Id",
        "SalePrice",
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "TotalHouseSF",
        "TotalLivingSF",
        "TotalBasementSF",
        "GarageCars",
        "GarageArea",
        "YearBuilt",
        "YearRemodAdd",
        "HouseAgeAtSale",
        "Neighborhood",
        "MSSubClass",
        "OverallCond",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
    ]

    available_columns = [
        column
        for column in important_columns
        if column in error_df.columns
    ]

    error_df = error_df[
        available_columns
    ]

    print("\n" + "=" * 80)
    print("HIGH-ERROR OBSERVATIONS")
    print("=" * 80)

    print(
        error_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()