# ============================================
# Gender Classification API
# ============================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch.
# This avoids downloading CUDA/NVIDIA packages.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    sentence-transformers

COPY app/__init__.py ./app/__init__.py
COPY app/gender_api.py ./app/gender_api.py

COPY models/gender_classifier.joblib ./models/gender_classifier.joblib

ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5002

EXPOSE 5002

CMD ["python", "-m", "app.gender_api"]