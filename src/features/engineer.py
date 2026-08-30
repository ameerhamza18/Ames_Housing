from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ( PROJECT_ROOT/ "data"/ "raw"/ "ames_housing.csv")
TARGET_COLUMN = "SalePrice"


def engineer_features(df: pd.DataFrame,) -> pd.DataFrame:
    df = df.copy()

    #Total house living area
    #-----------------------------
    area_columns = [ "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",]
    existing_area_columns = [ column for column in area_columns if column in df.columns ]
    if existing_area_columns:
        df["TotalHouseSF"] = ( df[existing_area_columns] .fillna(0).sum(axis=1))

    #Total above ground livinng area
    # ----------------------------
    above_ground_columns = ["GrLivArea",]
    if all(column in df.columns for column in above_ground_columns):
        df["TotalLivingSF"] = (df["GrLivArea"])

    #Total Basement area
    #-----------------------------
    if "TotalBsmtSF" in df.columns:
        df["TotalBasementSF"] = (df["TotalBsmtSF"])

    
    # Totral porch area
    #------------------------------
    porch_columns = [ "OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", "WoodDeckSF",]
    existing_porch_columns = [column for column in porch_columns if column in df.columns ]
    if existing_porch_columns:
        df["TotalPorchSF"] = ( df[existing_porch_columns] .fillna(0) .sum(axis=1))

 
    # Total bathrooms
    # ----------------------------
    bathroom_components = {"FullBath": 1.0,"HalfBath": 0.5,"BsmtFullBath": 1.0,"BsmtHalfBath": 0.5,}
    available_bathroom_columns = [ column for column in bathroom_components if column in df.columns]
    if available_bathroom_columns:
        total_bathrooms = 0
        for column in available_bathroom_columns:
            weight = bathroom_components[column]
            total_bathrooms += (df[column].fillna(0) * weight)

        df["TotalBathrooms"] = ( total_bathrooms)

    # Total rooms
    # --------------------------
    room_columns = ["TotRmsAbvGrd","BedroomAbvGr","FullBath",]
    if all( column in df.columns for column in room_columns):
        df["TotalFunctionalRooms"] = (df["TotRmsAbvGrd"].fillna(0) + df["BedroomAbvGr"].fillna(0)+ df["FullBath"].fillna(0))

    # House age at time of sale
    # ----------------------------
    if {"YearBuilt","YrSold",}.issubset(df.columns):
        df["HouseAgeAtSale"] = ( df["YrSold"] - df["YearBuilt"])

  
    # Years since remodeling
    if {"YearRemodAdd","YrSold", }.issubset(df.columns):
        df["YearsSinceRemodel"] = ( df["YrSold"] - df["YearRemodAdd"])

    
    # Garage age
    if { "GarageYrBlt", "YrSold",}.issubset(df.columns):
        df["GarageAgeAtSale"] = (df["YrSold"]- df["GarageYrBlt"])

  
    # Overall quality × living area
    if {"OverallQual","GrLivArea",}.issubset(df.columns):
        df["Quality_LivingArea"] = (df["OverallQual"]* df["GrLivArea"] )

   
    # Overall quality × house age
    if {"OverallQual","HouseAgeAtSale",}.issubset(df.columns):
        df["Quality_HouseAge"] = (df["OverallQual"]* df["HouseAgeAtSale"] )

 
    # Garage capacity × quality
    if { "GarageCars", "OverallQual",}.issubset(df.columns):
        df["Quality_GarageCapacity"] = ( df["OverallQual"] * df["GarageCars"])

    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate_engineered_features( original_df: pd.DataFrame, engineered_df: pd.DataFrame,) -> None:
  
    if len(original_df) != len(engineered_df):
        raise ValueError(
            "Feature engineering changed the number of rows."
        )

    missing_original_columns = [ column for column in original_df.columns if column not in engineered_df.columns]

    if missing_original_columns:
        raise ValueError( "Feature engineering removed existing columns: "
            f"{missing_original_columns}"
        )

    if TARGET_COLUMN not in engineered_df.columns:
        raise ValueError("Target column SalePrice is missing." )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print("AMES HOUSING — FEATURE ENGINEERING")
    print("=" * 80)

    print("\n[1/4] Loading dataset...")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError( f"Dataset not found: {RAW_DATA_PATH}")
    df = pd.read_csv(  RAW_DATA_PATH)
    print( f"Original shape: {df.shape}")


    # ------------------------------------------------------------------------
    print( "\n[2/4] Creating domain-informed features...")
    engineered_df = engineer_features( df)

  
    # ------------------------------------------------------------------------
    print( "\n[3/4] Validating engineered dataset...")
    validate_engineered_features( df, engineered_df, )

 
    # ------------------------------------------------------------------------
    original_features = set(df.columns)
    new_features = [ column for column in engineered_df.columns if column not in original_features]

    print( f"New features created: {len(new_features)}")
    print("\nNew features:")
    for feature in new_features:
        print(f"  - {feature}" )

    print(f"\nEngineered shape: "f"{engineered_df.shape}" )
    print( "\n[4/4] Feature engineering validation passed.")

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()