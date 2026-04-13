"""
Model prediction interface — DEPRECATED / DEAD CODE.

This file was the original RED-phase stub from TDD Cycle 0.
The real model implementation is ChurnModel in src/app/dependencies.py.

DO NOT import or instantiate ModelPredictor — it raises NotImplementedError.
It is retained here only to preserve the TDD commit history.
Removal decision: delete this file in a follow-up clean-up commit so the
git history shows the intentional progression from stub → real model.
"""


class ModelPredictor:
    """
    DEPRECATED: Original RED-phase stub. Not used in production.
    Real implementation: ChurnModel in src/app/dependencies.py.
    """

    def __init__(self, model_path: str):
        raise NotImplementedError(
            "ModelPredictor is a deprecated stub. "
            "Use ChurnModel from src.app.dependencies instead."
        )

    def predict(self, features: list[float]) -> float:
        raise NotImplementedError(
            "ModelPredictor is a deprecated stub. "
            "Use ChurnModel from src.app.dependencies instead."
        )
