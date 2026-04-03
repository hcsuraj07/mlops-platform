# =============================================================================
# serve.py
# PURPOSE: Load the registered model from MLflow and serve predictions
#          via FastAPI. This is how a trained model becomes a live API.
#
# FLOW:
#   FastAPI starts → loads model from MLflow registry → waits for requests
#   POST /predict  → validates input → runs model → returns prediction + probability
#   GET /health    → returns model version and status
#
# WHY LOAD FROM MLFLOW REGISTRY AND NOT JUST A FILE?
#   A file path like "models/model.pkl" is fragile — if you retrain and
#   overwrite the file, you lose the previous version. The MLflow registry
#   gives you versioned, named models. You can say "load production version"
#   and MLflow handles the rest. When you promote a new model to production,
#   the serving code doesn't change — just the registry pointer does.
# =============================================================================

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# MLFLOW SETUP
#
# Same tracking URI as train.py — must match so we can find the model.
# We load by model name + version. "1" means version 1 in the registry.
# In production you'd load by stage: "Production" or "Staging" instead.
# -----------------------------------------------------------------------------
mlflow.set_tracking_uri("mlruns")

MODEL_NAME = "credit-default-xgboost"
MODEL_VERSION = "1"

# -----------------------------------------------------------------------------
# LOAD MODEL AT STARTUP
#
# We load the model ONCE when the server starts, not on every request.
# WHY? Loading a model takes ~1-2 seconds. If you loaded it per request,
# every prediction would take 2+ seconds. Loading once at startup means
# predictions take milliseconds.
#
# mlflow.xgboost.load_model() pulls the model from the mlruns folder
# using the registered name and version.
# -----------------------------------------------------------------------------
print(f"Loading model: {MODEL_NAME} v{MODEL_VERSION}...")
model = mlflow.xgboost.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)
print("Model loaded successfully.")

app = FastAPI(
    title="Credit Default Prediction API",
    description="Predicts probability of credit card default using XGBoost",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
#
# Pydantic validates every incoming request automatically.
# If a required field is missing or wrong type, FastAPI returns 422.
#
# We use the same meaningful column names from train.py — not X1, X2...
# This makes the API self-documenting. Anyone hitting /docs immediately
# understands what each field means.
# -----------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    credit_limit: float
    gender: int           # 1=male, 2=female
    education: int        # 1=grad school, 2=university, 3=high school
    marriage: int         # 1=married, 2=single, 3=other
    age: int
    payment_status_sep: int   # -1=on time, 1-9=months delayed
    payment_status_aug: int
    payment_status_jul: int
    payment_status_jun: int
    payment_status_may: int
    payment_status_apr: int
    bill_amt_sep: float
    bill_amt_aug: float
    bill_amt_jul: float
    bill_amt_jun: float
    bill_amt_may: float
    bill_amt_apr: float
    payment_amt_sep: float
    payment_amt_aug: float
    payment_amt_jul: float
    payment_amt_jun: float
    payment_amt_may: float
    payment_amt_apr: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "credit_limit": 20000,
                "gender": 2,
                "education": 2,
                "marriage": 1,
                "age": 24,
                "payment_status_sep": 2,
                "payment_status_aug": 2,
                "payment_status_jul": -1,
                "payment_status_jun": -1,
                "payment_status_may": -2,
                "payment_status_apr": -2,
                "bill_amt_sep": 3913,
                "bill_amt_aug": 3102,
                "bill_amt_jul": 689,
                "bill_amt_jun": 0,
                "bill_amt_may": 0,
                "bill_amt_apr": 0,
                "payment_amt_sep": 0,
                "payment_amt_aug": 689,
                "payment_amt_jul": 0,
                "payment_amt_jun": 0,
                "payment_amt_may": 0,
                "payment_amt_apr": 0
            }]
        }
    }

class PredictionResponse(BaseModel):
    prediction: int           # 0=no default, 1=default
    probability: float        # probability of default (0.0 to 1.0)
    risk_level: str           # "low", "medium", "high" — business friendly
    model_version: str


# -----------------------------------------------------------------------------
# FEATURE ENGINEERING
#
# CRITICAL: must apply the EXACT same feature engineering as train.py.
# If train.py adds utilization_ratio but serve.py doesn't, the model
# receives 23 features instead of 26 and crashes immediately.
# In production, this logic lives in a shared preprocessing module
# imported by both train.py and serve.py — we'll refactor to that later.
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["utilization_ratio"] = df["bill_amt_sep"] / (df["credit_limit"] + 1)

    payment_cols = ["payment_status_sep", "payment_status_aug", "payment_status_jul",
                    "payment_status_jun", "payment_status_may", "payment_status_apr"]
    df["total_delays"] = df[payment_cols].apply(lambda x: (x > 0).sum(), axis=1)

    df["avg_payment_ratio"] = (
        df[["payment_amt_sep", "payment_amt_aug", "payment_amt_jul"]].mean(axis=1) /
        (df[["bill_amt_sep", "bill_amt_aug", "bill_amt_jul"]].mean(axis=1) + 1)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # convert request to dataframe — model expects a dataframe, not a dict
        input_df = pd.DataFrame([request.model_dump()])

        # apply same feature engineering as training
        input_df = engineer_features(input_df)

        # run prediction
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = int(probability >= 0.5)

        # convert probability to human-readable risk level
        # WHY ADD RISK LEVEL? Business teams don't think in probabilities.
        # "high risk" is more actionable than "0.73 probability".
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            model_version=MODEL_VERSION
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Credit Default Prediction API",
        "docs": "http://localhost:8001/docs",
        "health": "http://localhost:8001/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)