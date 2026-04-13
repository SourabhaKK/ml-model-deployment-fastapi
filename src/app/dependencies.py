"""
Dependency injection for model loading.
Production: loads churn RandomForestClassifier from joblib artifact.
"""
import logging
import threading
from typing import List

import joblib
import numpy as np

from src.app.config import settings

logger = logging.getLogger(__name__)


class ChurnModel:
    """
    Wraps the trained RandomForestClassifier from the churn prediction pipeline.
    Loaded from a joblib artifact dict produced by scripts/export_model.py in
    the customer-churn-prediction-ml repo.
    """

    def __init__(self) -> None:
        logger.info("Loading churn model from %s", settings.model_path)
        artifact = joblib.load(settings.model_path)
        self._model = artifact["model"]
        self.version: str = artifact["model_version"]
        self.feature_count: int = artifact["feature_count"]
        logger.info(
            "Churn model loaded — version=%s feature_count=%d",
            self.version, self.feature_count
        )

    def predict(self, features: List[float]) -> float:
        """
        Run inference. Returns prediction as Python float.
        sklearn returns numpy scalar — explicit cast prevents JSON serialisation errors.
        """
        X = np.array([features])           # shape (1, feature_count)
        return float(self._model.predict(X)[0])

    def predict_proba(self, features: List[float]) -> np.ndarray:
        """
        Return class probability array of shape (1, 2).
        Index [0][1] is the probability of churn (class 1).
        """
        X = np.array([features])           # shape (1, feature_count)
        return self._model.predict_proba(X)


# Singleton + thread-safe initialisation (F-06)
_model_instance = None
_model_lock = threading.Lock()


def get_model() -> ChurnModel:
    """Return the ChurnModel singleton, initialising it on first call."""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = ChurnModel()
    return _model_instance
