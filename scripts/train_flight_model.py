from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "travel_capstone_dataset" / "flights.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "flight_price_pipeline.joblib"


# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"
MLFLOW_EXPERIMENT = "Voyage Analytics - Flight Price Prediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ============================================================
# Main training workflow
# ============================================================


def main():
    print("=" * 70)
    print("Voyage Analytics - Flight Price Training")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate input dataset
    # --------------------------------------------------------

    print(f"Dataset path: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Flights dataset not found: {DATA_PATH}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Flights dataset is empty.")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_columns = [
        "from",
        "to",
        "flightType",
        "time",
        "distance",
        "agency",
        "date",
        "price",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_values = int(df[required_columns].isnull().sum().sum())

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values " "in required columns."
        )

    duplicate_rows = int(df.duplicated().sum())

    print(f"Duplicate rows: {duplicate_rows}")

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # --------------------------------------------------------
    # Feature engineering
    # Same logic as the production notebook
    # --------------------------------------------------------

    model_df = df.copy()

    model_df["year"] = model_df["date"].dt.year
    model_df["month"] = model_df["date"].dt.month
    model_df["day"] = model_df["date"].dt.day
    model_df["day_of_week"] = model_df["date"].dt.dayofweek

    # Remove identifiers and raw date
    model_df = model_df.drop(
        columns=[
            "travelCode",
            "userCode",
            "date",
        ]
    )

    X = model_df.drop(columns=["price"])
    y = model_df["price"]

    print("\nFeatures:")
    print(X.columns.tolist())

    print(f"Target: {y.name}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # --------------------------------------------------------
    # Preprocessing
    # Same structure as production notebook
    # --------------------------------------------------------

    categorical_features = [
        "from",
        "to",
        "flightType",
        "agency",
    ]

    numerical_features = [
        "time",
        "distance",
        "year",
        "month",
        "day",
        "day_of_week",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    # --------------------------------------------------------
    # Time-based evaluation split
    # Same strategy used earlier in the notebook
    # --------------------------------------------------------

    train_mask = df["date"].dt.year < 2023
    test_mask = df["date"].dt.year == 2023

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if X_train.empty:
        raise ValueError("Training set is empty.")

    if X_test.empty:
        raise ValueError("2023 test set is empty.")

    print("\nTime-based split:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples : {len(X_test):,}")

    # --------------------------------------------------------
    # Final production LightGBM model
    # Same parameters as notebook
    # --------------------------------------------------------

    final_lgbm = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", final_lgbm),
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining final LightGBM production model...")

    with mlflow.start_run(run_name="Airflow_Final_LightGBM_Training"):

        final_pipeline.fit(X_train, y_train)

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        y_pred = final_pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        r2 = r2_score(y_test, y_pred)

        print("\nModel Performance")
        print("-" * 40)
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.4f}")

        # ----------------------------------------------------
        # MLflow logging
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model": "LightGBM",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "validation_strategy": "Time-based split",
                "training_years": "<2023",
                "testing_year": 2023,
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
            }
        )

        mlflow.log_metrics(
            {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
            }
        )

        # Log the trained pipeline to MLflow
        mlflow.sklearn.log_model(
            final_pipeline,
            name="flight_price_pipeline",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor",
            ],
        )

        print("\nMLflow run logged successfully.")

        # ----------------------------------------------------
        # Save local production artifact
        # ----------------------------------------------------

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(final_pipeline, MODEL_PATH)

        print(f"Model saved to: {MODEL_PATH}")

    # --------------------------------------------------------
    # Reload verification
    # --------------------------------------------------------

    loaded_pipeline = joblib.load(MODEL_PATH)

    sample_predictions = loaded_pipeline.predict(X_test.head(5))

    print("\nSaved model reload verification:")
    print(sample_predictions)

    print("\nTraining workflow completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
