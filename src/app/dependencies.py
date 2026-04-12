"""
Dependency injection for model loading and other shared resources.
CYCLE 3 GREEN PHASE: Model dependency injection with singleton pattern.
"""
import os
import threading
from typing import List


class DummyModel:
    """
    Dummy model for testing dependency injection.
    Returns deterministic predictions without real ML logic.
    """
    # F-05: Expose model version for response tracing and rollback decisions
    version: str = os.getenv("MODEL_VERSION", "0.1.0-dummy")

    def __init__(self) -> None:
        # F-22: For real PyTorch models, call self.model.eval() here so dropout
        # and batch-norm layers switch to inference mode. Without it, the same
        # input returns different predictions on every call (non-deterministic).
        # Pair with torch.no_grad() inside predict() to avoid gradient tracking.
        pass

    def predict(self, features: List[float]) -> float:
        """
        Make a deterministic dummy prediction.

        Args:
            features: List of input features

        Returns:
            Deterministic prediction value (always 0.0)

        Note (F-21): Always cast the raw model output to Python float before
        returning. sklearn's model.predict() returns numpy.ndarray, not float.
        Failing to cast causes json serialisation to fail or Pydantic to coerce
        unpredictably. Pattern: return float(self.model.predict(X)[0])
        """
        # F-21: explicit float() cast — ensures a plain Python float is returned
        # even when a real model returns a numpy scalar or single-element array.
        return float(0.0)


# Singleton instance and lock for thread-safe initialisation
_model_instance = None
# F-06: threading.Lock ensures only one thread initialises the model,
# preventing double-instantiation under concurrent startup (double-checked locking pattern).
_model_lock = threading.Lock()


def get_model() -> DummyModel:
    """
    Get the model instance (singleton pattern).
    Model is loaded once and cached for reuse.

    Returns:
        DummyModel instance
    """
    global _model_instance

    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:  # re-check inside lock
                _model_instance = DummyModel()

    return _model_instance
