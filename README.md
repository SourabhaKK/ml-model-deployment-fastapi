# ML Model Deployment with FastAPI

> **Production-grade ML model serving API demonstrating Test-Driven Development (TDD), dependency injection, error handling, and comprehensive testing practices.**

[![Tests](https://img.shields.io/badge/tests-26%20passed-success)](https://github.com/SourabhaKK/ml-model-deployment-fastapi)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/SourabhaKK/ml-model-deployment-fastapi)
[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)](https://fastapi.tiangolo.com/)

## 🎯 Project Overview

This project showcases **production-ready ML engineering practices** through a FastAPI-based model serving API, built entirely using **Test-Driven Development (TDD)**. It demonstrates the complete journey from initial failing tests to a fully functional, production-ready API with comprehensive test coverage.

**Key Highlights:**
- ✅ **100% Test Coverage** - 26 tests across 5 testing categories
- ✅ **Strict TDD Methodology** - 5 complete RED-GREEN cycles
- ✅ **Production Patterns** - Dependency injection, error handling, validation
- ✅ **Clean Architecture** - Separation of concerns, testable design
- ✅ **End-to-End Testing** - Integration tests with minimal mocking

## 🚀 Why This Project Matters

This project demonstrates critical skills for **ML Engineering** and **Data Science** roles:

1. **Production ML Deployment** - Moving beyond notebooks to production-ready APIs
2. **Software Engineering Discipline** - TDD, clean code, proper testing
3. **API Design** - RESTful endpoints, validation, error handling
4. **Testing Expertise** - Unit tests, integration tests, mocking, dependency injection
5. **DevOps Readiness** - Structured for CI/CD, containerization, monitoring

## 📊 Technical Stack

- **Framework:** FastAPI (async, high-performance)
- **Validation:** Pydantic (type safety, schema validation)
- **Testing:** Pytest (26 tests, 100% coverage)
- **Architecture:** Dependency injection, singleton pattern
- **Error Handling:** Structured exceptions, HTTP status codes

## 🏗️ Architecture

### Project Structure
```
ml-model-deployment-fastapi/
├── src/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── routes.py         # API endpoints with error handling
│   │   ├── schemas.py        # Pydantic models with validation
│   │   ├── dependencies.py   # Dependency injection (singleton model)
│   │   └── config.py         # Configuration management
│   ├── model/
│   │   └── predictor.py      # Model interface
│   └── core/
│       └── exceptions.py     # Custom exceptions
├── tests/
│   ├── test_health.py              # Health endpoint tests
│   ├── test_predict_contract.py   # Prediction contract tests
│   ├── test_validation.py         # Input validation tests
│   ├── test_model_dependency.py   # Dependency injection tests
│   ├── test_error_handling.py     # Error handling tests
│   └── test_integration.py        # End-to-end integration tests
├── requirements.txt
└── pyproject.toml
```

### Design Patterns

- **Dependency Injection:** Model loaded once (singleton), injected into endpoints
- **Schema Validation:** Pydantic models for request/response validation
- **Error Handling:** Try-catch blocks with meaningful HTTP 500 responses
- **Separation of Concerns:** Routes, schemas, dependencies, and models separated

## 🔧 API Endpoints

### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "ok"
}
```

### Prediction
```http
POST /predict
Content-Type: application/json

{
  "features": [1.0, 2.0, 3.0, 4.0, 5.0]
}
```
**Response:**
```json
{
  "prediction": 0.0
}
```

**Validation:**
- ✅ Features must be a list of floats
- ✅ Features list cannot be empty
- ✅ Returns HTTP 422 for validation errors
- ✅ Returns HTTP 500 for prediction failures

## 🧪 Testing Strategy

### Test Coverage (26 Tests)

| Category | Tests | Description |
|----------|-------|-------------|
| **Health Endpoint** | 3 | Health check functionality |
| **Prediction Contract** | 4 | API contract verification |
| **Input Validation** | 5 | Schema validation, error handling |
| **Dependency Injection** | 4 | Model loading, singleton pattern |
| **Error Handling** | 4 | Exception handling, HTTP 500 responses |
| **Integration** | 6 | End-to-end flow with minimal mocking |

### TDD Cycles Completed

1. **Cycle 0:** Project structure & health endpoint
2. **Cycle 1:** Prediction endpoint contract
3. **Cycle 2:** Input validation with Pydantic
4. **Cycle 3:** Model dependency injection
5. **Cycle 4:** Error handling for prediction failures
6. **Cycle 5:** End-to-end integration testing

Each cycle followed strict RED → GREEN methodology:
- **RED:** Write failing tests defining requirements
- **GREEN:** Implement minimal code to pass tests
- **REFACTOR:** Improve code while maintaining green tests

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/SourabhaKK/ml-model-deployment-fastapi.git
cd ml-model-deployment-fastapi

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Start the server
uvicorn src.app.main:app --reload

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/test_integration.py -v
```

## 📈 Key Learnings & Best Practices

### 1. Test-Driven Development
- **Benefits:** Caught edge cases early, ensured requirements met, enabled confident refactoring
- **Approach:** Started with failing tests, implemented minimal solutions, refactored incrementally

### 2. Production Readiness
- **Error Handling:** All exceptions caught and converted to proper HTTP responses
- **Validation:** Pydantic schemas prevent invalid data from reaching the model
- **Dependency Injection:** Singleton pattern ensures efficient resource usage

### 3. API Design
- **RESTful:** Clear endpoints, proper HTTP methods and status codes
- **Documentation:** FastAPI auto-generates OpenAPI/Swagger docs
- **Type Safety:** Pydantic ensures type correctness at runtime

### 4. Testing Philosophy
- **Unit Tests:** Test individual components in isolation
- **Integration Tests:** Test complete flows with minimal mocking
- **Mocking Strategy:** Use FastAPI's dependency override for testability

## 🔮 Future Enhancements

This project is designed to be extended with:

- [ ] Real ML model integration (scikit-learn, PyTorch, TensorFlow)
- [ ] Model versioning and A/B testing
- [ ] Async prediction with background tasks
- [ ] Caching layer (Redis) for predictions
- [ ] Monitoring and logging (Prometheus, ELK stack)
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Authentication and rate limiting

## 💼 Skills Demonstrated

**For ML Engineering Roles:**
- Production ML deployment patterns
- API design for model serving
- Testing ML systems
- Error handling in production
- Dependency management

**For Data Science Roles:**
- Moving from notebooks to production
- Software engineering best practices
- API integration for ML models
- Testing and validation
- Production-ready code

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 👤 Author

**Sourabha K K**
- GitHub: [@SourabhaKK](https://github.com/SourabhaKK)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

**⭐ If you find this project helpful, please consider giving it a star!**

*This project demonstrates production-grade ML engineering practices suitable for enterprise ML deployment scenarios.*
