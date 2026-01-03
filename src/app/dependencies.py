"""
Dependency injection for model loading and other shared resources.
CYCLE 3 GREEN PHASE: Model dependency injection with singleton pattern.
"""
from typing import List


class DummyModel:
    """
    Dummy model for testing dependency injection.
    Returns deterministic predictions without real ML logic.
    """
    
    def predict(self, features: List[float]) -> float:
        """
        Make a deterministic dummy prediction.
        
        Args:
            features: List of input features
            
        Returns:
            Deterministic prediction value (always 0.0)
        """
        # Dummy prediction - deterministic output
        # No real ML model logic
        return 0.0


# Singleton instance of the model
_model_instance = None


def get_model() -> DummyModel:
    """
    Get the model instance (singleton pattern).
    Model is loaded once and cached for reuse.
    
    Returns:
        DummyModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        # Load model once at first call
        _model_instance = DummyModel()
    
    return _model_instance
