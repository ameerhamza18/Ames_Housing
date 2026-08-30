# Implementation Plan: Production Components for House Price ML

This plan outlines the steps to complete the productionization of the House Price ML pipeline, focusing on API serving, testing, containerization, monitoring, CI/CD, and experiment tracking.

## 1. Model Serving API
**Goal**: Provide a production-ready interface to generate house price predictions.

### Implementation
- **File to create**: `api/main.py`
- **Logic**:
    - Use **FastAPI** to build the web server.
    - **Model Lifecycle**: Load the trained Lasso model from `models/lasso_house_price_model.joblib` on startup to avoid reloading for every request.
    - **Request Validation**: Define a Pydantic model `HouseFeatures` containing all input features (based on the sample house in `src/inference/predict.py`).
    - **Endpoint**: Implement `POST /predict` which:
        1. Receives a JSON payload.
        2. Converts the payload to a pandas DataFrame.
        3. Calls `src.inference.predict.predict_price`.
        4. Returns the predicted price in a JSON response.
- **Verification**:
    - Run the API using `uvicorn api.main:app --reload`.
    - Use the built-in Swagger UI (`/docs`) to send a sample request and verify the output.

## 2. Automated Testing
**Goal**: Ensure robustness and prevent regressions in the data, feature, and inference pipelines.

### Implementation
- **Framework**: `pytest`
- **Files to create**:
    - `test/test_data.py`: Unit tests for `src/data/validate.py` (e.g., testing `validate_schema` with missing columns).
    - `test/test_features.py`: Unit tests for `src/features/preprocess.py` (e.g., verifying `identify_features` correctly splits types).
    - `test/test_inference.py`: Integration tests for `src/inference/predict.py` using a mock model or the saved joblib model.
    - `test/test_api.py`: API tests using `fastapi.testclient` to verify the `/predict` endpoint returns correct types and handles errors.
- **Verification**:
    - Execute `pytest test/` and ensure all tests pass.

## 3. Containerization
**Goal**: Ensure consistent deployment across different environments.

### Implementation
- **Files to create**:
    - `Dockerfile`: 
        - Base image: `python:3.11-slim`.
        - Install dependencies from `requirements.txt`.
        - Copy `src/`, `api/`, and `models/` into the image.
        - Set `PYTHONPATH=.` to allow imports from `src`.
        - Command: `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
    - `docker-compose.yml`:
        - Define a service `house-price-api` based on the Dockerfile.
        - Map port `8000:8000`.
- **Verification**:
    - Run `docker-compose up --build` and verify the API is reachable via `http://localhost:8000/predict`.

## 4. Model Monitoring
**Goal**: Track model performance and prediction distributions in production.

### Implementation
- **Files to create**:
    - `src/monitoring/monitor.py`: Implement a `PredictionMonitor` class.
        - **Logging**: Save each prediction, input features, and timestamp to a local file (e.g., `logs/predictions.csv`).
        - **Stats**: Methods to calculate the rolling mean of predicted prices and check for distribution shifts.
- **Integration**:
    - Instantiate `PredictionMonitor` in `api/main.py` and call its log method inside the `/predict` endpoint.
- **Verification**:
    - Send multiple requests to the API and verify that `logs/predictions.csv` is populated with the correct data.

## 5. CI/CD
**Goal**: Automate testing and linting on every code change.

### Implementation
- **File to create**: `.github/workflows/main.yml`
- **Logic**:
    - Trigger on `push` and `pull_request` to the `main` branch.
    - Steps:
        1. Checkout code.
        2. Setup Python environment.
        3. Install dependencies.
        4. Run linting (e.g., `flake8` or `black --check`).
        5. Run tests via `pytest`.
- **Verification**:
    - Trigger a GitHub Action run by pushing a dummy commit and verify the "All tests passed" check.

## 6. Experiment Tracking
**Goal**: Log hyperparameters and metrics for every model training run.

### Implementation
- **File to modify**: `src/models/final_model.py`
- **Logic**:
    - Integrate **MLflow**.
    - Wrap the main training loop in `with mlflow.start_run():`.
    - **Log Params**: Log `ALPHA`, `TEST_SIZE`, and `RANDOM_STATE`.
    - **Log Metrics**: Log the calculated `MAE`, `RMSE`, and `R2` scores.
    - **Log Model**: Use `mlflow.sklearn.log_model` to save the pipeline.
- **Verification**:
    - Run the training script and check the MLflow UI (`mlflow ui`) to see the recorded run and metrics.

## Sequencing & Dependencies
1. **Testing** $\rightarrow$ First, to ensure a stable baseline.
2. **API** $\rightarrow$ Second, to create the serving layer.
3. **Monitoring** $\rightarrow$ Integrated into the API.
4. **Containerization** $\rightarrow$ To package the API and its dependencies.
5. **CI/CD** $\rightarrow$ To automate the tests and container checks.
6. **Experiment Tracking** $\rightarrow$ Parallel to model refinement.
