# ==============================
# Travel MLOps Flask API
# ==============================

FROM python:3.12-slim

# Prevent Python from writing .pyc files
# and ensure logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application directory
WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy Flask application
COPY app ./app

# Copy local model artifact
COPY models ./models

# Flask configuration
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5001

# MLflow URI will be supplied at runtime
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Expose Flask port
EXPOSE 5001

# Start Flask application
CMD ["python", "-m", "app.app"]