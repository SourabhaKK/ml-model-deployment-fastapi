"""
Custom exception classes for the application.
"""


class ModelNotLoadedError(Exception):
    """Raised when attempting to use a model that hasn't been loaded."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


class InvalidInputError(Exception):
    """Raised when input validation fails."""
    pass
