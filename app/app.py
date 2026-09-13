# ==============================
# Flight Price Prediction API
# ==============================

from datetime import datetime

import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request

from .config import (
    DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
)

# ------------------------------
# Flask Application
# ------------------------------

app = Flask(__name__)


# ------------------------------
# Load Production Model
# ------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = mlflow.pyfunc.load_model(MODEL_URI)


# ------------------------------
# Required API Fields
# ------------------------------

REQUIRED_FIELDS = [
    "from",
    "to",
    "flightType",
    "time",
    "distance",
    "agency",
    "date",
]


# ------------------------------
# Health Check
# ------------------------------


@app.get("/health")
def health():
    return (
        jsonify(
            {
                "status": "healthy",
                "model": MODEL_NAME,
                "alias": MODEL_ALIAS,
            }
        ),
        200,
    )


# ------------------------------
# Prediction Endpoint
# ------------------------------


@app.post("/predict")
def predict():
    try:
        # --------------------------
        # Validate JSON body
        # --------------------------

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return (
                jsonify({"error": "Request body must contain a valid JSON object."}),
                400,
            )

        # --------------------------
        # Check required fields
        # --------------------------

        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]

        if missing_fields:
            return (
                jsonify(
                    {
                        "error": "Missing required fields.",
                        "missing_fields": missing_fields,
                    }
                ),
                400,
            )

        # --------------------------
        # Parse and validate date
        # --------------------------

        try:
            prediction_date = datetime.strptime(str(data["date"]), "%Y-%m-%d")
        except (ValueError, TypeError):
            return (
                jsonify(
                    {
                        "error": "Invalid date format.",
                        "message": "Use YYYY-MM-DD format.",
                    }
                ),
                400,
            )

        # --------------------------
        # Build model input
        # --------------------------

        input_data = {
            "from": data["from"],
            "to": data["to"],
            "flightType": data["flightType"],
            "time": float(data["time"]),
            "distance": float(data["distance"]),
            "agency": data["agency"],
            "year": prediction_date.year,
            "month": prediction_date.month,
            "day": prediction_date.day,
            "day_of_week": prediction_date.weekday(),
        }

        input_df = pd.DataFrame([input_data])

        # --------------------------
        # Model prediction
        # --------------------------

        prediction = model.predict(input_df)

        predicted_price = float(prediction[0])

        return (
            jsonify(
                {
                    "predicted_price": round(predicted_price, 2),
                    "model": MODEL_NAME,
                    "model_alias": MODEL_ALIAS,
                }
            ),
            200,
        )

    # --------------------------
    # Invalid numeric input
    # --------------------------

    except (TypeError, ValueError):
        return (
            jsonify(
                {
                    "error": "Invalid input data.",
                    "message": "time and distance must be numeric.",
                }
            ),
            400,
        )

    # --------------------------
    # Unexpected server error
    # --------------------------

    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "Prediction failed.",
                    "message": str(exc),
                }
            ),
            500,
        )


# ------------------------------
# Application Entry Point
# ------------------------------

if __name__ == "__main__":
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG,
    )
