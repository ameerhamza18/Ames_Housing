# 🏡 Ames Housing Price Predictor

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009485.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An end-to-end, production-grade machine learning system designed to predict residential house prices. This project demonstrates a complete ML Lifecycle: from raw data exploration and rigorous feature engineering to a containerized API deployment with real-time monitoring.

## 🚀 Impact & Performance

The project achieved a significant performance leap by transitioning from a naive baseline to a regularized model with log-transformed targets.

| Model Phase | Target Scale | Regularization | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Raw | None | \$20,102 | \$60,821 | 0.5177 |
| **Intermediate** | Log | None | \$16,095 | \$25,806 | 0.9132 |
| **Production** | Log | Ridge/Lasso | **\$15,919** | **\$25,033** | **0.9183** |

**Key Insight**: Implementing a log-transformation on the target variable (`SalePrice`) reduced the RMSE by ~58%, effectively handling the price skewness and reducing the impact of luxury home outliers.

## 🌟 Professional Engineering Highlights

### 1. Leakage-Safe ML Pipeline
Implemented a custom preprocessing pipeline that ensures no data leakage occurs between training and validation sets. All transformations are fitted on the training split and applied to the test split.

### 2. Domain-Informed Feature Engineering
Moved beyond raw data by creating synthetically engineered features that correlate strongly with market value:
- **Total Living Area**: Aggregated basement and above-ground square footage.
- **Quality Composite**: Weighted scoring of overall material and construction quality.
- **Temporal Decay**: Calculated the age of the home at the time of sale to capture depreciation.

### 3. Production-Grade Serving Layer
- **High-Performance API**: Built with **FastAPI**, utilizing asynchronous endpoints for low-latency predictions.
- **Strict Validation**: Every request is validated via **Pydantic** schemas to prevent "garbage-in, garbage-out" scenarios.
- **Frontend Integration**: A responsive **Tailwind CSS** dashboard providing a seamless user experience for non-technical stakeholders.

### 4. Reliability & Observability
- **Model Monitoring**: Integrated a `PredictionMonitor` that logs every production inference, enabling the detection of data drift over time.
- **Automated Testing**: Pytest suite covering both the feature engineering logic and the API response integrity.
- **Containerization**: Fully Dockerized architecture ensuring consistency from local development to cloud production.

## 🏗️ System Architecture

`Raw Data` $\rightarrow$ `Feature Engineering` $\rightarrow$ `Lasso/Ridge Regression` $\rightarrow$ `FastAPI` $\rightarrow$ `Web UI`

- **Core ML (`/src`)**: Data cleaning, feature engineering, and model optimization.
- **Serving Layer (`/api`)**: Loads the serialized `.joblib` model and provides a RESTful interface.
- **Presentation Layer (`/api/static`)**: A client-side web application for interactive price estimation.

## 🚦 Getting Started

### Prerequisites
- Python 3.12+
- Docker (Optional)

### Local Setup

```bash
# Clone and enter the project
git clone https://github.com/your-username/house-price-ml.git
cd house-price-ml

# Environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Launch the production server
python -m api.main
```

Visit **[http://localhost:8000](http://localhost:8000)** to interact with the model.

## 🐳 Deployment Strategy

### Docker (Recommended)
```bash
# Build the image
docker build -t house-price-app .

# Run the container
docker run -p 8000:8000 house-price-app
```

### Cloud Options
- **PaaS (Render/Railway)**: Direct GitHub integration $\rightarrow$ Start command: `python -m api.main`.
- **Managed Kubernetes (EKS/GKE)**: Deploy the Docker image as a scalable microservice.

## 📁 Project Structure

```text
house-price-ml/
├── api/                # Serving Layer (FastAPI + Frontend)
│   ├── main.py         # Application entry point
│   ├── schemas.py      # Pydantic validation logic
│   └── static/         # Modern Web UI (Tailwind CSS)
├── data/               # Data Lake
│   ├── raw/            # Original dataset (ames_housing.csv)
│   └── processed/      # Cleaned & transformed data
├── models/             # Model Registry
│   └── lasso_house_price_model.joblib # Production weights
├── reports/            # Analytics & Audit Trail
│   ├── figures/        # Model diagnostics & EDA plots
│   └── model_comparison.csv # Benchmarking results
├── src/                # Engineering Core
│   ├── data/           # Ingestion, validation & quality checks
│   ├── eda/            # Exploratory data analysis logic
│   ├── features/       # Feature engineering pipeline
│   ├── inference/      # Prediction logic
│   ├── models/         # Model training & hyperparameter tuning
│   └── monitoring/     # Production observability
├── test/               # Pytest suite (Unit & Integration)
├── Dockerfile          # Container definition
└── requirements.txt    # Dependency manifest
```

---
**Developed as a showcase of Machine Learning Engineering (MLE) best practices.**
