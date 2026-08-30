from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = (PROJECT_ROOT / "data" / "raw" / "ames_housing.csv")
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
INVESTIGATION_REPORT_PATH = (REPORTS_DIR / 
                             "feature_investigation.txt")


# Configuration
TARGET_COLUMN = "SalePrice"
ID_COLUMN ="Id"

# Dataset loading
def load_dataset() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:{RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset loaded successfully:{df.shape}")
    return df


# Features Semantics
def identify_feature_types(df:pd.DataFrame) -> dict:
    numerical_features = df.select_dtypes(include="number").columns.tolist()
    categorical_fetaures = df.select_dtypes(exclude="number").columns.tolist()

    if ID_COLUMN  in numerical_features:
        numerical_features.remove(ID_COLUMN)

    semantic_categorical_features = ["MSSubClass", "MsSold"]
    for column in semantic_categorical_features:
        if column in numerical_features:
            numerical_features.remove(column)
            categorical_fetaures.append(column)

    return{
        "numerical":numerical_features,
        "categorical": categorical_fetaures,
        "semantic_categorical":semantic_categorical_features
    }


# Missing Values Investigation
def investigate_missing_values(df:pd.DataFrame,)-> pd.DataFrame:
    results =[]

    for column in df.columns:
        missing_count= int(df[column].isna().sum())
        if missing_count == 0:
            continue
        missing_percentage = (missing_count / len(df) * 100)

        results.append({
            "feature":column,
            "missing_count":missing_count,
            "missing_percentage": round(missing_percentage,2)
        })

    return (pd.DataFrame(results)
            .sort_values("missing_percentage", ascending=False))




# Numerical Skewness
def analyze_skewness(df:pd.DataFrame,
                      numerical_features:list) -> pd.DataFrame:
    results =[]
    for column in numerical_features:
        series = df[column].dropna()
        if series.empty:
            continue
        results.append({
            "feature":column,
            "skewness": float(series.skew()),
            "absolute_skewness":abs(float(series.skew()))
        })

    return(pd.DataFrame(results).sort_values(
        "absolute_skewness", ascending=False
    ))




# Target Relationship
def analyze_target_relationships(df: pd.DataFrame,numerical_features: list,) -> pd.DataFrame:
    results = []

    for column in numerical_features:
        correlation = df[[column, TARGET_COLUMN] ].corr(numeric_only=True).iloc[0, 1]
        results.append({"feature": column,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            }
        )

    return ( pd.DataFrame(results) .sort_values(
            "absolute_correlation",ascending=False, ))




# Feature-feature correlation
def analyze_multicollinearity(df: pd.DataFrame,numerical_features: list,) -> pd.DataFrame:
    correlation_matrix = ( df[numerical_features] .corr() .abs())
    pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1,len(correlation_matrix.columns),):
            feature_a = (correlation_matrix.columns[i])
            feature_b = (  correlation_matrix.columns[j] )

            correlation = correlation_matrix.iloc[ i, j]
            if correlation >= 0.70:
                pairs.append({
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "absolute_correlation": round(float(correlation),4,),
                    }
                )

    return (pd.DataFrame(pairs).sort_values("absolute_correlation",ascending=False, )
        if pairs
        else pd.DataFrame( columns=[
                "feature_a",
                "feature_b",
                "absolute_correlation",
            ]
        )
    )




# Categorical cardinality
def analyze_categorical_features( df: pd.DataFrame, categorical_features: list,) -> pd.DataFrame:
    results = []
    for column in categorical_features:
        results.append({
            "feature": column,
                "unique_categories": int( df[column].nunique(  dropna=True  )),
                "missing_count": int( df[column].isna().sum()  ),
            }
        )

    return ( pd.DataFrame(results) .sort_values(
            "unique_categories",
            ascending=False,
        )
    )




# Categorical target analysis
def analyze_categorical_target_relationship( df: pd.DataFrame, categorical_features: list,) -> dict:
    results = {}
    for column in categorical_features:
        grouped = ( df.groupby( column, dropna=False,)[TARGET_COLUMN]
            .agg( [ "count", "mean", "median", ] )
            .sort_values("mean",ascending=False, )
        )

        results[column] = grouped

    return results




# Target vs important numerical features
def plot_target_relationships( df: pd.DataFrame,target_relationships: pd.DataFrame,) -> None:
    FIGURES_DIR.mkdir( parents=True, exist_ok=True,)

    top_features = ( target_relationships.head(6)["feature"].tolist())
    for feature in top_features:

        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=df,x=feature,y=TARGET_COLUMN,)

        plt.title(f"{feature} vs {TARGET_COLUMN}")
        plt.tight_layout()

        filename = (f"{feature}_vs_saleprice.png")
        plt.savefig( FIGURES_DIR / filename, dpi=150,)
        plt.close()




# Target outlier investigation
def investigate_target_outliers(df: pd.DataFrame,) -> dict:
    target = df[TARGET_COLUMN]

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(target < lower_bound)| (target > upper_bound)]

    return {
         "q1": float(q1),
        "q3": float(q3),
        "lower_bound": float( lower_bound ),
        "upper_bound": float(upper_bound),
        "outlier_count": int( len(outliers)),
        "outlier_percentage": round( len(outliers) / len(df) * 100,2,),
    }



# Log transformation diagnostic
def analyze_target_transformation( df: pd.DataFrame,) -> dict:
   original_skewness = float(df[TARGET_COLUMN].skew())

   log_target = np.log1p(df[TARGET_COLUMN])
   log_skewness = float(log_target.skew())

   return {
        "original_skewness": original_skewness,
        "log1p_skewness": log_skewness,
    }




# Generate investigation report
def generate_report(feature_types: dict,missing_values: pd.DataFrame,
    skewness: pd.DataFrame,target_relationships: pd.DataFrame,
    multicollinearity: pd.DataFrame,categorical_cardinality: pd.DataFrame,
    target_outliers: dict,target_transformation: dict,) -> None:
    
    REPORTS_DIR.mkdir(parents=True,exist_ok=True,)

    with open( INVESTIGATION_REPORT_PATH, "w",encoding="utf-8",) as file:
        file.write("AMES HOUSING — FEATURE INVESTIGATION\n")

        file.write("=" * 80)
        file.write("\n\n")

        file.write("FEATURE SEMANTICS\n")
        file.write("-" * 80)
        file.write("\n")
        file.write(f"Numerical features: "f"{len(feature_types['numerical'])}\n")
        file.write( f"Categorical features: "f"{len(feature_types['categorical'])}\n" )
        file.write( "Semantic categorical features:\n")
        for feature in feature_types[ "semantic_categorical"]:
            file.write( f"  - {feature}\n")
        file.write("\n")

    

        file.write("MISSING VALUE INVESTIGATION\n")
        file.write("-" * 80)
        file.write("\n")
        file.write(missing_values.to_string(index=False ))
        file.write("\n\n")


        file.write("NUMERICAL SKEWNESS\n")
        file.write("-" * 80)
        file.write("\n")
        file.write(skewness.head(20).to_string(index=False) )
        file.write("\n\n")

    
        file.write( "NUMERICAL FEATURES vs SALEPRICE\n")
        file.write("-" * 80)
        file.write("\n")
        file.write( target_relationships.head(20).to_string(index=False))
        file.write("\n\n")

      

        file.write( "HIGH FEATURE-FEATURE CORRELATIONS\n" )
        file.write("-" * 80)
        file.write("\n")
        if multicollinearity.empty:
            file.write( "No feature pairs exceeded the threshold.\n")
        else:
            file.write(  multicollinearity.to_string(index=False) )
        file.write("\n\n")

      

        file.write( "CATEGORICAL FEATURE CARDINALITY\n" )
        file.write("-" * 80)
        file.write("\n")
        file.write(categorical_cardinality.to_string( index=False))
        file.write("\n\n")

     
        file.write( "TARGET OUTLIER INVESTIGATION\n" )
        file.write("-" * 80)
        file.write("\n")
        for key, value in target_outliers.items():
            file.write( f"{key}: {value}\n"   )
        file.write("\n")

      
        file.write("TARGET TRANSFORMATION DIAGNOSTIC\n" )
        file.write("-" * 80)
        file.write("\n")
        file.write(f"Original skewness: "f"{target_transformation['original_skewness']:.6f}\n")
        file.write(  f"log1p skewness: " f"{target_transformation['log1p_skewness']:.6f}\n")
        file.write("\n")




# Main
def main() -> None:

    print("=" * 80)
    print("AMES HOUSING — DEEP FEATURE INVESTIGATION")
    print("=" * 80)
    df = load_dataset()

    print("\n[1/8] Identifying feature semantics...")
    feature_types = identify_feature_types(df)
    print( f"Numerical: " f"{len(feature_types['numerical'])}" )
    print( f"Categorical: " f"{len(feature_types['categorical'])}" )
    print( "Semantic categorical:", feature_types["semantic_categorical"], )

   

    print("\n[2/8] Investigating missing values...")
    missing_values = investigate_missing_values(df)
    print(missing_values.head(10).to_string(index=False )  )



    print("\n[3/8] Analyzing numerical skewness...")
    skewness = analyze_skewness(df,feature_types["numerical"],)
    print(skewness.head(10).to_string( index=False ) )

 

    print("\n[4/8] Analyzing target relationships...")
    target_relationships = ( analyze_target_relationships( df, feature_types["numerical"],))
    print(target_relationships.head(10).to_string( index=False ) )

  
    print("\n[5/8] Investigating feature-feature correlation...")
    multicollinearity = (analyze_multicollinearity(df,feature_types["numerical"],))
    print( multicollinearity.head(15).to_string(index=False))

 
    print("\n[6/8] Analyzing categorical cardinality...")
    categorical_cardinality = (analyze_categorical_features( df, feature_types["categorical"],))
    print(categorical_cardinality.head(15).to_string(index=False ))

  

    print("\n[7/8] Investigating target outliers...")
    target_outliers = (investigate_target_outliers(df))
    print(f"Target outliers: "f"{target_outliers['outlier_count']}")
    print(f"Target outlier percentage: "f"{target_outliers['outlier_percentage']}%")



    print("\n[8/8] Comparing target skewness...")
    target_transformation = (analyze_target_transformation(df))
    print( f"Original skewness: "f"{target_transformation['original_skewness']:.6f}")
    print(f"log1p skewness: "f"{target_transformation['log1p_skewness']:.6f}")


    
    print( "\nGenerating target relationship plots...")
    plot_target_relationships(df,target_relationships,)


    print("\nGenerating investigation report...")

    generate_report(feature_types,missing_values,skewness,
        target_relationships,multicollinearity,categorical_cardinality,
        target_outliers,target_transformation,)

    print("\n" + "=" * 80)
    print("DEEP EDA COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print( f"Report: {INVESTIGATION_REPORT_PATH}")
    print(f"Figures: {FIGURES_DIR}" )


if __name__ == "__main__":
    main()