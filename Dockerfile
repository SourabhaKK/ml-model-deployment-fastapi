FROM python:3.11-slim

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
