from datetime import datetime
from pathlib import Path
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path("/opt/airflow/project")

DATA_PATH = PROJECT_ROOT / "travel_capstone_dataset" / "flights.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "flight_price_pipeline.joblib"

TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "train_flight_model.py"


# ============================================================
# Tasks
# ============================================================


def validate_data():
    print("=" * 60)
    print("Validating flight dataset...")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Flights dataset not found: {DATA_PATH}")

    print(f"Dataset found: {DATA_PATH}")


def train_model():
    print("=" * 60)
    print("Starting real flight-price model training...")
    print("=" * 60)

    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAINING_SCRIPT}")

    result = subprocess.run(
        [
            "python",
            str(TRAINING_SCRIPT),
        ],
        check=True,
        capture_output=False,
    )

    print(
        f"Training script completed successfully "
        f"with exit code {result.returncode}."
    )


def verify_model():
    print("=" * 60)
    print("Verifying saved flight-price model...")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Production model not found: {MODEL_PATH}")

    print(f"Production model exists: {MODEL_PATH}")
    print("Model verification completed successfully.")


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="flight_price_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=[
        "travel",
        "mlops",
        "flight-price",
    ],
    description=(
        "Flight price ML pipeline with "
        "dataset validation, model training, "
        "MLflow logging, and model verification."
    ),
) as dag:

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    verify = PythonOperator(
        task_id="verify_model",
        python_callable=verify_model,
    )

    validate >> train >> verify
