# ML Model Deployment — FastAPI Inference Service

![CI](https://github.com/SourabhaKK/ml-model-deployment-fastapi/actions/workflows/ci.yml/badge.svg)

Production-grade FastAPI service for serving ML models with
authentication, rate limiting, circuit breaking, and async inference.

---

## Architecture

```
Request
│
├─► API Key Middleware (X-API-Key)
│
├─► Rate Limit Middleware (per-IP sliding window)
│
└─► POST /predict
      │
      ├─► Pydantic v2 input validation
      │     (type, NaN/Inf guard, feature count check)
      │
      ├─► Circuit Breaker (CLOSED / OPEN / HALF_OPEN)
      │
      ├─► async run_in_executor → model.predict()
      │     (5s timeout, MemoryError guard)
      │
      └─► PredictionResponse (prediction, confidence, model_version)
```

---

## Production Features

| Feature | Detail |
|---|---|
| Authentication | API key via `X-API-Key` header (opt-in via `API_KEY` env var) |
| Rate limiting | Per-IP sliding window (configurable via `RATE_LIMIT_PER_MINUTE`) |
| Circuit breaker | CLOSED / OPEN / HALF\_OPEN states (opt-in via `CIRCUIT_BREAKER_ENABLED`) |
| Async inference | `run_in_executor` prevents event loop blocking; 5s timeout |
| Request tracing | UUID `X-Request-ID` on every response |
| Readiness probe | `/ready` returns 503 until model is loaded |
| Structured logging | JSON-format logs parseable by Datadog / CloudWatch |
| Pydantic v2 | Strict schema validation with NaN/Inf guards and shape checking |
| Confidence scores | `predict_proba` output returned alongside binary prediction |
| Model versioning | `MODEL_VERSION` env var surfaced in every response |

---

## Quick Start

### Run locally
```bash
pip install -r requirements.txt
uvicorn src.app.main:app --reload
```

### Run with Docker
```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

### Run with docker-compose
```bash
docker compose up --build
```

### Run tests
```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description | Auth required |
|---|---|---|---|
| `POST` | `/predict` | Run model inference | If `API_KEY` set |
| `GET` | `/health` | Liveness check | No |
| `GET` | `/ready` | Readiness check (503 if model not loaded) | No |

### POST /predict — Request
```json
{
  "features": [1.0, 2.0, 3.0, 4.0]
}
```

### POST /predict — Response
```json
{
  "prediction": 0.0,
  "confidence": 0.87,
  "model_version": "1.0.0"
}
```

> `confidence` is omitted when the model does not implement `predict_proba`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_VERSION` | `0.1.0` | Version tag returned in all responses |
| `MODEL_PATH` | `models/churn_model.pkl` | Path to serialised model file |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_KEY` | `""` | If non-empty, enforces `X-API-Key` header |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP request cap (0 = disabled) |
| `CIRCUIT_BREAKER_ENABLED` | `false` | Enable circuit breaker |
| `EXPECTED_FEATURE_COUNT` | `0` | 0 = disabled; N = enforces exact feature count |

Copy `.env.example` to `.env` and edit values for local development.

---

## Ecosystem

This service is part of an end-to-end ML system:

| Layer | Repo | Role |
|---|---|---|
| Model training | [customer-churn-prediction](https://github.com/SourabhaKK/customer-churn-prediction) | Trains and exports the serialised model served here |
| Drift monitoring | [ml-model-monitoring-drift-detection](https://github.com/SourabhaKK/ml-model-monitoring-drift-detection) | Monitors prediction distributions for data drift |

---

## Project Structure

```
src/
├── app/
│   ├── main.py          # App factory, middleware, lifespan
│   ├── routes.py        # /predict, /health, /ready endpoints
│   ├── schemas.py       # Pydantic v2 request/response models
│   ├── dependencies.py  # Model loading and DI factory
│   └── config.py        # pydantic-settings centralised config
tests/
├── test_health.py
├── test_predict_contract.py
├── test_validation.py
├── test_model_dependency.py
├── test_error_handling.py
└── test_integration.py
```
