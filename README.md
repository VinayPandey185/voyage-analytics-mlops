# ✈️ Voyage Analytics — End-to-End MLOps Travel Intelligence Platform

> **Voyage Analytics** is an end-to-end Machine Learning and MLOps platform that demonstrates how travel-related ML solutions can be developed, automated, tracked, deployed, scaled, and consumed through a production-oriented architecture.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-blue)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-Orchestration-017CEE?logo=apacheairflow)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployment-326CE5?logo=kubernetes)](https://kubernetes.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)](https://streamlit.io/)

---

## 📌 Project Overview

Voyage Analytics combines **machine learning, data exploration, feature engineering, workflow orchestration, experiment tracking, model management, REST API serving, containerization, Kubernetes deployment, and autoscaling** into a single travel analytics platform.

The platform provides three major ML/data capabilities:

- ✈️ **Flight Price Prediction** — estimates flight prices using route, flight type, duration, distance, agency, and travel date.
- 🏨 **Hotel Recommendation** — recommends available hotel/travel records based on destination, trip duration, budget, and dataset availability.
- 👤 **Gender Classification** — provides a separate classification workflow and API as part of the project's ML capabilities.
- 📊 **Travel Dataset Analytics** — explores and works with flight, hotel, and user datasets.

The project demonstrates the complete journey from **raw data and experimentation to production-style model serving and scalable deployment**.

### End-to-End Flow

```text
Datasets
   ↓
EDA & Data Validation
   ↓
Feature Engineering
   ↓
Model Training / Recommendation Logic
   ↓
MLflow Experiment Tracking
   ↓
MLflow Model Registry
   ↓
Production Model Alias
   ↓
Flask REST API
   ↓
Docker Container
   ↓
Kubernetes Deployment
   ↓
Horizontal Pod Autoscaling
   ↓
Streamlit Application
```

---

## 🎯 Project Objectives

### 1. Build Travel Machine Learning Solutions

Develop multiple ML/data workflows for different travel-related use cases:

- Regression for flight price prediction
- Classification for gender prediction
- Data-driven recommendation for hotel selection

### 2. Perform Data Exploration & Feature Engineering

Analyze raw travel data and transform it into features suitable for machine learning.

For flight price prediction, the system uses:

```text
Categorical:
- from
- to
- flightType
- agency

Numerical:
- time
- distance
- year
- month
- day
- day_of_week
```

The travel date is transformed into year, month, day, and day-of-week features.

### 3. Manage Models with MLflow

MLflow is used for experiment tracking and model lifecycle management.

The production flight model is registered as:

```text
Voyage_Flight_Price_Model
```

and served through the:

```text
champion
```

alias.

This allows the application to load the designated model version through MLflow instead of hardcoding a local model version.

### 4. Automate ML Workflows

Apache Airflow is used to orchestrate important pipeline stages such as data validation, training, and model verification.

### 5. Serve Predictions through REST APIs

The Flask backend exposes model-serving endpoints including:

```text
GET  /health
POST /predict
```

The `/predict` endpoint performs inference using the MLflow-managed flight price model.

### 6. Containerize the Application

Docker packages the application and dependencies into reproducible runtime environments.

### 7. Deploy and Scale with Kubernetes

The Flask inference service is deployed on Docker Desktop Kubernetes.

The deployment demonstrates:

- Pods
- Deployments
- Services
- Multiple replicas
- Horizontal Pod Autoscaling (HPA)

Configured scaling:

```text
Minimum replicas: 2
Maximum replicas: 5
CPU target: 70%
```

### 8. Provide an Interactive User Interface

Streamlit provides the user-facing travel analytics application for flight price prediction and hotel recommendation.

---

## 💡 Why This Is an MLOps Project

A traditional ML workflow may stop at:

```text
Dataset → Training → Model → Prediction
```

Voyage Analytics extends this into an operational ML lifecycle:

```text
Dataset
   ↓
Data Validation
   ↓
Training Pipeline
   ↓
Experiment Tracking
   ↓
Model Registry
   ↓
Model Alias
   ↓
REST API
   ↓
Docker
   ↓
Kubernetes
   ↓
HPA Scaling
   ↓
Streamlit Application
```

The focus is therefore not only on model development, but also on:

- Reproducibility
- Model lifecycle management
- Workflow automation
- Deployment
- API-based inference
- Containerization
- Scalability
- Operational troubleshooting

---

## 🎯 Practical Use Case

### ✈️ Flight Price Prediction

A user provides information such as:

```text
From: Mumbai
To: Delhi
Flight Type: Economy
Duration: 120 minutes
Distance: 1,400 km
Agency: Air India
Travel Date: 20 September 2026
```

The Streamlit application sends the request to the Flask API. The backend retrieves the registered `champion` model through MLflow and returns an estimated flight price.

### 🏨 Hotel Recommendation

A user selects:

```text
Destination
Trip Duration
Maximum Budget
Number of Recommendations
```

The application searches the available hotel/travel records and returns suitable recommendations based on the dataset.

---

## 📊 Dataset Details

The project uses three travel datasets stored under:

```text
travel_capstone_dataset/
```

### ✈️ Flight Dataset

**File:** `travel_capstone_dataset/flights.csv`

| Attribute | Details |
|---|---|
| Records | **271,888** |
| Columns | 10 |
| Target | `price` |
| Problem Type | Regression |
| Model | LightGBM |

Columns:

```text
travelCode
userCode
from
to
flightType
price
time
distance
agency
date
```

The model derives the following date features:

```text
year
month
day
day_of_week
```

The final production pipeline contains:

```text
OneHotEncoder
+
Numerical feature handling
+
LightGBM Regressor
```

### 🏨 Hotel Dataset

**File:** `travel_capstone_dataset/hotels.csv`

| Attribute | Details |
|---|---|
| Records | **40,552** |
| Columns | 8 |
| Location field | `place` |
| Approach | Filtering / ranking |

Columns:

```text
travelCode
userCode
name
place
days
price
total
date
```

The recommendation workflow considers destination, trip duration, maximum budget, and available hotel/package records.

### 👤 User Dataset

**File:** `travel_capstone_dataset/users.csv`

The user dataset provides user-level travel information associated with travel records. It supports user/travel analysis and provides a foundation for future personalization and segmentation.

---

## 📈 Dataset Scale & Coverage

| Dataset | Records | Primary Use |
|---|---:|---|
| ✈️ Flights | **271,888** | Flight price prediction |
| 🏨 Hotels | **40,552** | Hotel recommendation |
| 👤 Users | User-level travel records | User/travel analysis |

### Location Coverage

The datasets contain a large number of records but a fixed set of destinations.

**Flights:**

- 9 unique origin locations
- 9 unique destination locations

**Hotels:**

- 9 unique destinations

Hotel destinations include:

```text
Florianopolis (SC)
Salvador (BH)
Natal (RN)
Aracaju (SE)
Recife (PE)
Sao Paulo (SP)
Campo Grande (MS)
Rio de Janeiro (RJ)
Brasilia (DF)
```

The application uses locations actually present in the datasets rather than inventing unsupported destinations.

---

## 🧠 Machine Learning Approach

### Flight Price Prediction

Flight price prediction is implemented as a supervised regression workflow.

The production model uses a Scikit-learn pipeline:

```text
ColumnTransformer
├── Categorical
│   └── OneHotEncoder(handle_unknown="ignore")
│
└── Numerical
    └── Passthrough
        ↓
LightGBM Regressor
```

Model configuration includes:

```text
learning_rate = 0.05
n_estimators = 300
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

### Hotel Recommendation

Hotel recommendation is a **data-driven filtering and ranking workflow**, rather than a separately trained recommendation model.

```text
Destination
    ↓
Trip Duration
    ↓
Maximum Budget
    ↓
Available Records
    ↓
Recommended Hotels / Packages
```

### Gender Classification

The project also contains a gender classification model and API workflow. The trained model artifact is maintained under:

```text
models/gender_classifier.joblib
```

---

## 🔄 MLOps Lifecycle

```text
┌──────────────────────┐
│    Travel Datasets   │
│ Flights / Hotels /   │
│ Users                │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Apache Airflow       │
│ Validation /         │
│ Training / Checks    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ MLflow               │
│ Tracking / Registry  │
│ Model Versioning     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ champion Model Alias │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Flask REST API       │
│ /health / /predict   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Docker               │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Kubernetes           │
│ Deployment + Service │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ HPA                  │
│ 2 → 5 replicas       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Streamlit UI         │
└──────────────────────┘
```

---

## 🧪 MLflow Model

The production flight model is registered in MLflow as:

```text
Model: Voyage_Flight_Price_Model
Alias: champion
Status: READY
```

The deployed Flask service loads the model through the MLflow Model Registry rather than bundling the model directly into the Streamlit application.

---

## ☸️ Kubernetes Deployment

The Flask prediction API is containerized and deployed to Kubernetes.

Key deployment components include:

```text
Deployment: travel-mlops-flask
Service:    travel-mlops-flask
Type:       NodePort
Port:       5001
HPA:        CPU-based autoscaling
Min Pods:   2
Max Pods:   5
```

Health endpoint:

```http
GET /health
```

Prediction endpoint:

```http
POST /predict
```

Example successful response:

```json
{
  "model": "Voyage_Flight_Price_Model",
  "model_alias": "champion",
  "predicted_price": 1250.50
}
```

---

## 🐳 Docker

The Flask service is packaged as a Docker image and deployed locally through Docker Desktop Kubernetes.

The container exposes port `5001` and connects to the MLflow tracking/model registry service through the Docker host.

---

## 🔁 CI/CD

A Jenkins pipeline is included in the repository for automating the project workflow and productionization steps.

The overall MLOps lifecycle is:

```text
Code
 ↓
Build
 ↓
Validate
 ↓
Train
 ↓
Track with MLflow
 ↓
Register Model
 ↓
Deploy
 ↓
Verify
```

---

## 🖥️ Running the Application Locally

### 1. Activate the virtual environment

```powershell
cd "C:\Users\VINAY\Desktop\Labmentix Projects\Travel_MLops_Major_Project"
.\.venv\Scripts\Activate.ps1
```

### 2. Start MLflow

```powershell
mlflow ui --host 0.0.0.0 --port 5000 --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:5000"
```

Keep this terminal running.

### 3. Verify Kubernetes

```powershell
kubectl get nodes
kubectl get pods -l app=travel-mlops-flask
kubectl get hpa travel-mlops-flask
```

### 4. Start the Flask port-forward

```powershell
kubectl port-forward svc/travel-mlops-flask 5001:5001
```

Keep this terminal running.

### 5. Verify the API

```powershell
Invoke-WebRequest http://localhost:5001/health -UseBasicParsing
```

### 6. Start Streamlit

In another terminal:

```powershell
cd "C:\Users\VINAY\Desktop\Labmentix Projects\Travel_MLops_Major_Project"
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

### 6. Start Airflow

In another terminal:

```powershell
cd "C:\Users\VINAY\Desktop\Labmentix Projects\Travel_MLops_Major_Project"
.\.venv\Scripts\Activate.ps

docker compose -f .\airflow\docker-compose.yml up -d

Then check: 

docker compose -f .\airflow\docker-compose.yml ps

Check whether port 8080 is responding: 

Test-NetConnection 127.0.0.1 -Port 8080
```
---

## 📓 Experimentation Notebooks

The repository contains three notebooks:

```text
notebooks/
├── 01_flight_price_prediction.ipynb
├── 02_gender_classification.ipynb
└── 03_travel_recommendation.ipynb
```

### `01_flight_price_prediction.ipynb`

Covers the flight price prediction workflow including:

- Dataset exploration
- Data preprocessing
- Feature engineering
- Model experimentation
- Regression evaluation

### `02_gender_classification.ipynb`

Covers:

- Classification data preparation
- Model experimentation
- Evaluation
- Model artifact generation

### `03_travel_recommendation.ipynb`

Covers:

- Travel/hotel data exploration
- Destination analysis
- Recommendation logic
- Result analysis

The notebooks provide the experimentation layer, while the production-oriented application uses the trained artifacts and deployment architecture.

---

## 🚀 Project Highlights

| Area | Implementation |
|---|---|
| Frontend | Streamlit |
| Hotel Recommendation | Pandas-based recommendation engine |
| Flight Prediction | LightGBM regression pipeline |
| Experiment / Model Management | MLflow |
| REST API | Flask |
| Workflow Orchestration | Apache Airflow |
| Containerization | Docker |
| Deployment | Kubernetes (Docker Desktop) |
| Autoscaling | Kubernetes HPA |
| CI/CD | Jenkins |
| Data Processing | Pandas / Scikit-learn |

---


## 🏗️ High-Level System Architecture

```text
                         ┌─────────────────────┐
                         │   Travel Datasets   │
                         │                     │
                         │ Flights / Hotels /  │
                         │ Users               │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Apache Airflow   │
                         │                     │
                         │ Validation          │
                         │ Training            │
                         │ Verification        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       MLflow        │
                         │                     │
                         │ Experiments         │
                         │ Artifacts           │
                         │ Model Registry      │
                         │ champion Alias      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Flask API      │
                         │                     │
                         │ /health             │
                         │ /predict            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Docker Container  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────┐
                 │       Kubernetes Cluster         │
                 │                                  │
                 │ Deployment → Pods → Service      │
                 │              ↓                   │
                 │             HPA                  │
                 └────────────────┬─────────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │    Streamlit UI     │
                         │                     │
                         │ Flight Prediction   │
                         │ Hotel Recommendation│
                         └─────────────────────┘
```

---

## ⚠️ Key Challenges & Solutions

### 1. MLflow Model Registry Connectivity

The Flask API initially failed while loading:

```text
models:/Voyage_Flight_Price_Model@champion
```

because the Docker container could not successfully communicate with the host MLflow server.

### 2. MLflow Host Validation

Requests from Docker initially returned:

```text
403 Forbidden
Invalid Host header - possible DNS rebinding attack detected
```

The MLflow server was configured with an appropriate allowed-host configuration for Docker Desktop communication.

### 3. Kubernetes CrashLoopBackOff

The Flask pods initially entered:

```text
CrashLoopBackOff
```

The Kubernetes logs identified the MLflow model-loading failure. After correcting MLflow connectivity/security configuration and restarting the deployment, the Flask pods became healthy.

### 4. Horizontal Pod Autoscaling

The inference service uses Kubernetes HPA to demonstrate automatic horizontal scaling:

```text
Minimum: 2 replicas
Maximum: 5 replicas
CPU target: 70%
```

### 5. Dataset Location Coverage

The datasets contain many records but only a fixed set of destinations. The UI therefore uses the actual dataset values.

### 6. Local Resource Constraints

Running Kubernetes, Airflow, PostgreSQL, MLflow, Docker, Flask, and Streamlit simultaneously can create significant local resource usage. Components were validated incrementally to keep the development environment stable.

---

## ⚙️ Technology Stack

| Layer | Technology |
|---|---|
| Programming | Python |
| UI | Streamlit |
| API | Flask |
| ML | Scikit-learn + LightGBM |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Workflow Orchestration | Apache Airflow |
| Database | PostgreSQL |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Autoscaling | Kubernetes HPA |
| CI/CD Configuration | Jenkins |
| Development | Jupyter Notebook / VS Code |

---

## 🚀 API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "alias": "champion",
  "model": "Voyage_Flight_Price_Model",
  "status": "healthy"
}
```

### Flight Prediction

```http
POST /predict
```

The endpoint receives flight features and returns an estimated price together with the registered model name and alias.

---

## ☸️ Kubernetes Operations

Useful commands:

```powershell
kubectl get nodes
kubectl get pods -l app=travel-mlops-flask
kubectl get hpa travel-mlops-flask
kubectl rollout status deployment/travel-mlops-flask
```

Example deployment configuration:

```text
Deployment: travel-mlops-flask
Service: travel-mlops-flask
Replicas: 2–5
CPU target: 70%
```

---

## 📁 Project Structure

```text
Travel_MLops_Major_Project/
│
├── app/
│   ├── app.py
│   ├── config.py
│   └── gender_api.py
│
├── airflow/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── dags/
│
├── docker/
│   └── Dockerfile
│
├── gender/
│   ├── Dockerfile
│   └── .dockerignore
│
├── kubernetes/
│   ├── deployment.yaml
│   ├── gender-classifier.yaml
│   └── service.yaml
│
├── models/
│   ├── flight_price_pipeline.joblib
│   └── gender_classifier.joblib
│
├── notebooks/
│   ├── 01_flight_price_prediction.ipynb
│   ├── 02_gender_classification.ipynb
│   └── 03_travel_recommendation.ipynb
│
├── screenshots/
│   ├── 01_airflow_pipeline.png
│   ├── 02_flight_prediction.png
│   ├── 03_hotel_recommendation.png
│   ├── 04_dataset_information.png
│   ├── 05_mlflow_model_registry.png
│   └── 06_kubernetes_hpa.png
│
├── scripts/
│   └── train_flight_model.py
│
├── travel_capstone_dataset/
│   ├── flights.csv
│   ├── hotels.csv
│   └── users.csv
│
├── streamlit_app.py
├── requirements.txt
├── Jenkinsfile
├── README.md
└── Voyage_Analytics_Project_Report.pdf
```

---

## 📸 Screenshots

### Airflow Pipeline

![Airflow Pipeline](screenshots/01_airflow_pipeline.png)

### Flight Price Prediction

![Flight Prediction](screenshots/02_flight_prediction.png)

### Hotel Recommendation

![Hotel Recommendation](screenshots/03_hotel_recommendation.png)

### Dataset Information

![Dataset Information](screenshots/04_dataset_information.png)

### MLflow Model Registry

![MLflow Model Registry](screenshots/05_mlflow_model_registry.png)

### Kubernetes HPA

![Kubernetes HPA](screenshots/06_kubernetes_hpa.png)

---

## 📄 Project Report

A detailed technical report is included in the repository:

**`Voyage_Analytics_Project_Report.pdf`**

The report contains:

- Project objectives
- Problem statement
- Dataset details
- Architecture
- ML workflow
- MLOps lifecycle
- Airflow workflow
- MLflow model management
- Docker deployment
- Kubernetes deployment
- HPA configuration
- Screenshots
- Challenges and solutions
- Project outcomes

---

## 📌 Important Dataset & Model Limitations

### Dataset Coverage

The supplied datasets contain a large number of records but represent a limited geographical area.

Therefore, the platform should be considered a **dataset-driven travel analytics and MLOps demonstration system**, rather than a live worldwide travel marketplace.

### Flight Price Predictions

Predicted prices are **model estimates based on historical dataset patterns**. They are not live airline prices and should not be interpreted as real-time market quotations.

### Hotel Recommendations

Hotel recommendations are generated from the available historical hotel/travel records in the dataset. They are not live hotel inventory or live booking availability.

---

## ⭐ Project Highlights

- End-to-end MLOps lifecycle
- **271K+ flight records**
- **40K+ hotel records**
- Three travel datasets
- Three experimentation notebooks
- Flight price regression
- Hotel recommendation workflow
- Gender classification
- MLflow experiment tracking
- MLflow Model Registry
- `champion` production alias
- Apache Airflow orchestration
- Flask REST API
- Docker containerization
- Kubernetes deployment
- Horizontal Pod Autoscaling
- Streamlit interactive UI
- Production-oriented troubleshooting
- Detailed technical project report

---

## 👨‍💻 Author

### Vinay Pandey

**MCA | Python | Machine Learning | MLOps | React.js**

📌 **GitHub:** [VinayPandey185](https://github.com/VinayPandey185)

💼 **LinkedIn:** [Vinay Pandey](https://www.linkedin.com/in/vinay-pandey/)

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

**Repository:** [voyage-analytics-mlops](https://github.com/VinayPandey185/voyage-analytics-mlops)
