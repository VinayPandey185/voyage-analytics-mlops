# ============================================
# Gender Classification API
# ============================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# --------------------------------------------
# Paths
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "gender_classifier.joblib"


# --------------------------------------------
# Flask Application
# --------------------------------------------

app = Flask(__name__)


# --------------------------------------------
# Load Gender Model Bundle
# --------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Gender model not found: {MODEL_PATH}")

model_bundle = joblib.load(MODEL_PATH)

embedding_model = model_bundle["embedding_model"]
pca = model_bundle["pca"]
company_encoder = model_bundle["company_encoder"]
age_scaler = model_bundle["age_scaler"]
classifier = model_bundle["classifier"]

CLASSES = model_bundle.get(
    "classes",
    list(classifier.classes_),
)


# --------------------------------------------
# Health Check
# --------------------------------------------


@app.get("/health")
def health():
    return (
        jsonify(
            {
                "status": "healthy",
                "model": "gender_classifier",
                "classes": CLASSES,
            }
        ),
        200,
    )


# --------------------------------------------
# Gender Prediction
# --------------------------------------------


@app.post("/predict-gender")
def predict_gender():
    try:
        data = request.get_json(silent=True)

        # Validate JSON object
        if not isinstance(data, dict):
            return (
                jsonify(
                    {"error": ("Request body must contain " "a valid JSON object.")}
                ),
                400,
            )

        # Required fields
        required_fields = [
            "name",
            "company",
            "age",
        ]

        missing_fields = [field for field in required_fields if field not in data]

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

        # Validate values
        name = str(data["name"]).strip()
        company = str(data["company"]).strip()

        if not name:
            return (
                jsonify({"error": "Name cannot be empty."}),
                400,
            )

        if not company:
            return (
                jsonify({"error": "Company cannot be empty."}),
                400,
            )

        try:
            age = float(data["age"])
        except (TypeError, ValueError):
            return (
                jsonify({"error": "Age must be numeric."}),
                400,
            )

        # ------------------------------------
        # Name embedding
        # ------------------------------------

        name_embedding = embedding_model.encode(
            [name],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # ------------------------------------
        # PCA transformation
        # ------------------------------------

        name_pca = pca.transform(name_embedding)

        # ------------------------------------
        # Company encoding
        # ------------------------------------

        company_features = company_encoder.transform(
            pd.DataFrame(
                {
                    "company": [company],
                }
            )
        )

        # ------------------------------------
        # Age scaling
        # ------------------------------------

        age_features = age_scaler.transform(
            pd.DataFrame(
                {
                    "age": [age],
                }
            )
        )

        # ------------------------------------
        # Final feature matrix
        # ------------------------------------

        final_features = np.hstack(
            [
                name_pca,
                company_features,
                age_features,
            ]
        )

        # ------------------------------------
        # Prediction
        # ------------------------------------

        prediction = classifier.predict(final_features)

        predicted_gender = str(prediction[0])

        return (
            jsonify(
                {
                    "predicted_gender": predicted_gender,
                    "model": "gender_classifier",
                }
            ),
            200,
        )

    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "Gender prediction failed.",
                    "message": str(exc),
                }
            ),
            500,
        )


# --------------------------------------------
# Application Entry Point
# --------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5002,
        debug=False,
    )
