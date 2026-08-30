import pandas as pd


results = pd.DataFrame(
    [
        {
            "Model": "Linear Regression - Raw Target",
            "MAE": 20102.59,
            "RMSE": 60821.62,
            "R2": 0.5177,
        },
        {
            "Model": "Linear Regression - Log Target",
            "MAE": 16095.57,
            "RMSE": 25806.36,
            "R2": 0.9132,
        },
        {
            "Model": "Ridge Regression - Log Target - Alpha 0.01",
            "MAE": 15919.20,
            "RMSE": 25033.86,
            "R2": 0.9183,
        },
    ]
)


results = results.sort_values( by="RMSE").reset_index(drop=True)

results.insert( 0, "Rank", range(1, len(results) + 1))


print("=" * 80)
print("MODEL EXPERIMENT LEADERBOARD")
print("=" * 80)

print(results.to_string( index=False))


results.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nLeaderboard saved to:")
print("reports/model_comparison.csv")