"""
conftest.py — session-scoped test fixtures.

ROOT CAUSE FIX (CI): churn_model.joblib is gitignored and absent in CI.
Any test that resolves Depends(get_model) without a dependency_override, or
calls get_model() directly, triggers ChurnModel() → joblib.load(...) →
FileNotFoundError.

FIX STRATEGY: Before any test runs, pre-load a MockChurnModel into the
module-level singleton (_model_instance). get_model() checks
  if _model_instance is None
and returns immediately when it is not — joblib.load() is never reached.

Tests that need specific model behaviour keep their own
  app.dependency_overrides[get_model] = lambda: mock
Those overrides operate at the FastAPI DI layer and are unaffected by this
fixture. When those tests call app.dependency_overrides.clear(), the
_model_instance singleton remains set, so subsequent requests continue to
use MockChurnModel rather than trying to re-load the real artifact.
"""
import numpy as np
import pytest

import src.app.dependencies as _deps


class MockChurnModel:
    """
    Lightweight stand-in for ChurnModel. Matches the interface contract:
      - .predict(features: List[float]) -> float  (returns 0.0, binary class)
      - .predict_proba(features: List[float]) -> np.ndarray  (shape (1, 2))
      - .version: str
      - .feature_count: int
    """

    version: str = "1.0.0-test"
    feature_count: int = 51

    def predict(self, features):
        return 0.0

    def predict_proba(self, features):
        # probability of class 0 = 0.8, class 1 (churn) = 0.2
        return np.array([[0.8, 0.2]])


_MOCK_MODEL = MockChurnModel()


@pytest.fixture(autouse=True, scope="session")
def install_mock_model():
    """
    Pre-load MockChurnModel into the singleton slot for the entire test
    session. This is the only change needed to make all 28 tests pass in CI
    without the real .joblib artifact.

    Runs before the first test; tears down after the last test.
    """
    _deps._model_instance = _MOCK_MODEL
    yield
    _deps._model_instance = None
