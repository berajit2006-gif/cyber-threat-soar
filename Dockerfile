# Base Python slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install build dependencies for XGBoost/SHAP C++ extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained artifacts
COPY . .

# Automatically train models if artifacts are missing
RUN python -c "import os; from train_pipeline import run_training_pipeline; os.path.exists('artifacts/xgb_model.joblib') or run_training_pipeline()"

# Expose target port
EXPOSE 8000

# Start production server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
