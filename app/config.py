# ==============================
# Flask Application Configuration
# ==============================

import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

MODEL_NAME = os.getenv("MODEL_NAME", "Voyage_Flight_Price_Model")

MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")

FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))

DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
