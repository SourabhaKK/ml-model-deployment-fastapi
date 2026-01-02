# ML Model Deployment with FastAPI

Production-grade ML model serving API built with Test-Driven Development (TDD).

## Project Status

**Current Phase:** 🔴 RED (Cycle 0)

This project follows strict TDD methodology:
- ✅ Project structure initialized
- ✅ API contracts defined via failing tests
- ⏳ Implementations pending (GREEN phase)

## Features (Planned)

- RESTful API for ML model predictions
- Input validation with Pydantic
- Health check endpoint
- Structured error handling
- Production-ready configuration

## Project Structure

```
ml-model-deployment-fastapi/
├── src/
│   ├── app/              # FastAPI application
│   │   ├── main.py       # App entry point
│   │   ├── schemas.py    # Pydantic models
│   │   ├── routes.py     # API routes
│   │   ├── dependencies.py
│   │   └── config.py
│   ├── model/            # ML model interface
│   │   └── predictor.py
│   └── core/             # Core utilities
│       └── exceptions.py
├── tests/                # Test suite
│   ├── test_health.py
│   ├── test_predict_contract.py
│   └── test_validation.py
├── requirements.txt
└── pyproject.toml
```

## API Endpoints (Planned)

### Health Check
```
GET /health
Response: {"status": "ok"}
```

### Prediction
```
POST /predict
Request: {"features": [1.0, 2.0, 3.0, 4.0]}
Response: {"prediction": 0.85, "model_version": "1.0.0"}
```

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (currently failing - RED phase)
pytest tests/ -v
```

## Testing Philosophy

This project uses TDD:
1. **RED:** Write failing tests that define contracts
2. **GREEN:** Implement minimal code to pass tests
3. **REFACTOR:** Improve code while keeping tests green

Current status: All tests intentionally fail (RED phase).

## License

MIT License - See LICENSE file for details.
