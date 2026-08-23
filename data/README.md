# Dataset Documentation

## 1. Dataset

### Name

Ames Housing — House Prices: Advanced Regression Techniques

### Domain

Residential Real Estate

### Problem Type

Supervised Learning → Regression

### Objective

Predict the sale price of a residential property from its available
property characteristics.

---

## 2. Observation

Each row represents a residential property sale in Ames, Iowa.

---

## 3. Target Variable

| Column | Type | Description |
|---|---|---|
| SalePrice | Numerical | Sale price of the property in USD |

The target variable is continuous, making this a regression problem.

---

## 4. Dataset Characteristics

Expected training dataset:

- Rows: 1,460
- Original columns: 81
- Predictive features: 79
- Identifier: 1
- Target: 1

The dataset contains both numerical and categorical variables.

---

## 5. Feature Categories

### Numerical Features

Examples:

- LotArea
- OverallQual
- OverallCond
- YearBuilt
- YearRemodAdd
- GrLivArea
- TotalBsmtSF
- GarageArea
- WoodDeckSF
- OpenPorchSF

### Categorical Features

Examples:

- MSZoning
- Street
- Neighborhood
- HouseStyle
- RoofStyle
- Exterior1st
- Exterior2nd
- Foundation
- Heating
- SaleType
- SaleCondition

Some categorical variables represent ordered quality levels and
therefore require special consideration during preprocessing.

---

## 6. Data Quality Risks

The dataset will be investigated for:

- Missing values
- Duplicate records
- Invalid values
- Outliers
- Skewed numerical distributions
- High-cardinality categorical features
- Multicollinearity
- Inconsistent data types
- Potential leakage

No cleaning decisions will be made before performing data profiling.

---

## 7. Identifier

`Id` is an identifier and is not expected to be used as a predictive
feature.

We will verify this during data validation and exploratory analysis.

---

## 8. Target Handling

The target distribution will be investigated during EDA.

We will evaluate whether a transformation such as a logarithmic
transformation is appropriate based on the data and model assumptions.

We will not apply transformations blindly.

---

## 9. Data Splitting Policy

The dataset will be divided into:

- Training set
- Validation strategy through cross-validation
- Final holdout test set

The test set must remain isolated from model development decisions.

Any preprocessing operation that learns parameters from data must be
fitted using training data only.

---

## 10. Data Leakage Policy

The project must prevent information from validation or test data
leaking into the training process.

Examples of operations that must be fitted only on training data:

- Imputation
- Scaling
- Encoding where applicable
- Feature selection
- Target-dependent transformations

Scikit-learn pipelines will eventually be used to enforce this
separation.

---

## 11. Reproducibility

All experiments should be reproducible.

We will maintain:

- Fixed random seeds where appropriate
- Versioned dependencies
- Explicit configuration
- Recorded model parameters
- Recorded evaluation metrics
- Versioned model artifacts

---

## 12. Source

The dataset originates from the Ames Housing dataset compiled by
Dean De Cock and is widely used for regression and machine-learning
education.

The project will document the original dataset source and licensing/
usage information when the raw data is added to the repository.

---

## 13. Important Principle

The raw dataset must never be silently modified.

Pipeline:

Raw Data
    ↓
Validation
    ↓
Cleaning / Transformation
    ↓
Processed Data
    ↓
Modeling

Every significant transformation should be reproducible through code.