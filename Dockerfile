# Base image pinned by digest (tag: python:3.11-slim); Dependabot keeps it current.
FROM python:3.11-slim@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Bake model artifact into image (churn_model.joblib, ~5MB)
COPY models/ ./models/

EXPOSE 8000

# Run as non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
