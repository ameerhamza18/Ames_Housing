import requests
import json

url = "http://localhost:8000/predict"
house_data = {
    "Id": 9999, "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 70, "LotArea": 8500, "Street": "Pave", "Alley": None, "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl", "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "2Story", "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2005, "YearRemodAdd": 2005, "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "MasVnrArea": 100, "ExterQual": "Gd", "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "Gd", "BsmtFinType1": "GLQ", "BsmtFinSF1": 600, "BsmtFinType2": "Unf", "BsmtFinSF2": 0, "BsmtUnfSF": 400, "TotalBsmtSF": 1000, "Heating": "GasA", "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr", "1stFlrSF": 1000, "2ndFlrSF": 800, "LowQualFinSF": 0, "GrLivArea": 1800, "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "Gd", "TotRmsAbvGrd": 7, "Functional": "Typ", "Fireplaces": 1, "FireplaceQu": "Gd", "GarageType": "Attchd", "GarageYrBlt": 2005, "GarageFinish": "Fin", "GarageCars": 2, "GarageArea": 500, "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y", "WoodDeckSF": 100, "OpenPorchSF": 50, "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": None, "Fence": None, "MiscFeature": None, "MiscVal": 0, "MoSold": 6, "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal"
}

try:
    response = requests.post(url, json=house_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
