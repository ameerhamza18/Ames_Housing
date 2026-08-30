from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = (PROJECT_ROOT / "data" /"raw" / "ames_housing.csv")

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"
TEST_SIZE = 0.20
RANDOM_STATE = 42


SEMANTIC_CATEGORICAL_FEATURES =["MSSubClass", "MoSold"]


# Load dataset
def load_data()->pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:{RAW_DATA_PATH}"
        )
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset loaded: {df.shape}")
    return df


# Identify Features
def identify_features(df:pd.DataFrame)-> tuple[list[str], list[str]]:
    numerical_features = (df.select_dtypes(include="number").columns.tolist())
    categorical_features = (df.select_dtypes(exclude="number").columns.tolist())

    if TARGET_COLUMN in numerical_features:
        numerical_features.remove(ID_COLUMN)
    if ID_COLUMN in categorical_features:
        categorical_features.remove(ID_COLUMN)

    for column in SEMANTIC_CATEGORICAL_FEATURES:
        numerical_features.remove(column)
        categorical_features.append(column)

    return numerical_features, categorical_features


#Build Numerical Pipeline
def build_numerical_pipeline()-> Pipeline:
    numerical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"),),
               ("scaler", StandardScaler(),),
               ]
    )

    return numerical_pipeline

#Build Categorical Pipeline
def build_categorical_pipleine()->Pipeline:
    categorical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Missing"),),
               ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True),),
               ]
    )
    return categorical_pipeline


#Build Complete Preprocessor
def build_preprocessor(numerical_fetaures:list[str],
                       categorical_faetures:list[str],)->ColumnTransformer:
    numerical_pipeline =(build_numerical_pipeline())
    categorical_pipeline = (build_categorical_pipleine())
    preprocessor = ColumnTransformer(
        transformers=[("numerical", numerical_pipeline,numerical_fetaures),
                      ("categorical", categorical_pipeline, categorical_faetures,),
                      ],  remainder="drop")
    return preprocessor



# Train Test Split
def split_data(df:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame,
                                       pd.Series, pd.Series,]:
    X = df.drop(columns=[TARGET_COLUMN])
    y =df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test =(train_test_split(X,y,test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE))
    return(X_train, X_test, y_train, y_test)


# Fit Preprocessor
def fit_preprocessor(X_train:pd.DataFrame)-> ColumnTransformer:
    numerical_features,categorical_features=(identify_features(X_train))
    preprocessor = build_preprocessor(numerical_features,
                                      categorical_features)
    preprocessor.fit(X_train)
    return preprocessor


# Transform Data
def transform_data(preprocessor:ColumnTransformer,
                   X_train:pd.DataFrame, X_test:pd.DataFrame):
    X_train_processed =(preprocessor.transform(X_train))
    X_test_processed = (preprocessor.transform(X_test))
    return(X_train_processed, X_test_processed)




# Main
def main()-> None:
    print("="*80)
    print("AMES Housninng - Data Preprocessing")
    print("="*80)

    print("\n[1/5] Loading Dataset......")
    df = load_data()

    print("\n[2/5] Identifying features types.......")
    numerical_features, categorical_fetaures =(identify_features(df))
    print(f"Numerical Features:"
          f"{len(numerical_features)}")
    print(f"Categorical Features:"
          f"{len(categorical_fetaures)}")
    print("Semantic categorical feature:",SEMANTIC_CATEGORICAL_FEATURES)


    print("\n[3/5] Creating train-test-split.......")
    (X_train , X_test, y_train , y_test) = split_data(df)
    print(f"Training samples:{len(X_train)}")
    print(f"Testing Samples:{len(X_test)}")


    print("\n[4/5] Fitting Preprocessing pipeline.......")
    preprocessor = fit_preprocessor(X_train=X_train)
    print("Preprocessor fitted successfullt")


    print("\n[5/5] Transforming train-test fetaures......")
    (X_train_processed, X_test_processed)= transform_data(
        preprocessor, X_train, X_test
    )
    print(f"Processed training shape:{X_train_processed.shape}")
    print(f"Processed testing shape:{X_test_processed.shape}")

    print("="*80)
    print("Preprocessing Completed Successfully")
    print("="*80)





if __name__ == "__main__":
    main()
