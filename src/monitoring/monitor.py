import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

class PredictionMonitor:
    """
    Basic monitor to log house price predictions and track drift.
    """
    def __init__(self, log_file: str = "predictions_log.csv"):
        self.log_path = Path(__file__).resolve().parents[2] / "reports" / log_file
        self._init_log()

    def _init_log(self):
        """Initialize the log file with headers if it doesn't exist."""
        if not self.log_path.exists():
            df = pd.DataFrame(columns=["timestamp", "predicted_price", "features"])
            df.to_csv(self.log_path, index=False)

    def log_prediction(self, price: float, features: Dict[str, Any]):
        """Log a single prediction event."""
        new_entry = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_price": price,
            "features": json.dumps(features)
        }])
        new_entry.to_csv(self.log_path, mode='a', header=False, index=False)

    def get_statistics(self):
        """Calculate basic statistics over the logged predictions."""
        df = pd.read_csv(self.log_path)
        if df.empty:
            return {}

        return {
            "count": len(df),
            "mean_price": df["predicted_price"].mean(),
            "median_price": df["predicted_price"].median(),
            "std_price": df["predicted_price"].std(),
            "min_price": df["predicted_price"].min(),
            "max_price": df["predicted_price"].max()
        }
