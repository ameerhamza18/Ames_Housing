import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from api.schemas import HouseFeatures, PredictionResponse
from src.inference.predict import load_model, predict_price
from src.monitoring.monitor import PredictionMonitor

app = FastAPI(
    title="House Price Prediction API",
    description="Production API for predicting residential house prices using the Ames Housing dataset.",
    version="1.0.0"
)

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("api/static/index.html")

# Singletons for the loaded model and monitor
MODEL = None
MONITOR = None

@app.on_event("startup")
async def startup_event():
    """Load the ML model and monitoring system on startup."""
    global MODEL, MONITOR
    try:
        MODEL = load_model()
        MONITOR = PredictionMonitor()
        print("Model and Monitor loaded successfully.")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise RuntimeError("Could not initialize production system on startup.")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if MODEL is not None:
        return {"status": "healthy", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict", response_model=PredictionResponse)
async def predict(house: HouseFeatures):
    """
    Predict the sale price of a house based on provided features.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to dictionary and then to DataFrame
        # Using by_alias=True to handle fields like '1stFlrSF'
        house_dict = house.model_dump(by_alias=True)
        house_df = pd.DataFrame([house_dict])

        # Generate prediction
        price = predict_price(MODEL, house_df)

        # Log prediction for monitoring
        if MONITOR:
            MONITOR.log_prediction(price, house_dict)

        return PredictionResponse(predicted_price=price)
    except Exception as e:

        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
