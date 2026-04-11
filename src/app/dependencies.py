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

    def predict(self, features: List[float]) -> float:
        """
        Make a deterministic dummy prediction.

        Args:
            features: List of input features

        Returns:
            Deterministic prediction value (always 0.0)
        """
        return 0.0


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
