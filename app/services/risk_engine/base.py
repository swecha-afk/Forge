"""Abstract Base Class and Factory for Hot-Swappable Disease Risk Models.

All predictors (Baseline XGBoost, Calibrated Triad Ensemble, etc.) inherit from
BaseDiseasePredictor and return the immutable StandardizedRiskOutput contract.
"""

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.swarm_schemas import StandardizedRiskOutput, WeatherTelemetryInput

# Global registry for hot-swappable models
PREDICTOR_REGISTRY: dict[str, type["BaseDiseasePredictor"]] = {}


def register_predictor(name: str):
    """Decorator to register model architectures in PREDICTOR_REGISTRY."""

    def decorator(cls):
        PREDICTOR_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class BaseDiseasePredictor(ABC):
    """Abstract contract ensuring 100% hot-swappable disease risk models."""

    model_name: str = "base_predictor"
    version: str = "1.0.0"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDiseasePredictor":
        """Fits the underlying statistical / ML model."""

    @abstractmethod
    def predict_and_explain(
        self, telemetry: WeatherTelemetryInput
    ) -> StandardizedRiskOutput:
        """Runs forward inference and computes exact feature attribution (TreeSHAP)."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persists model weights and calibration parameters to disk."""

    @abstractmethod
    def load(self, filepath: str) -> "BaseDiseasePredictor":
        """Loads model weights and calibration parameters from disk."""


def get_predictor(
    name: str = "baseline_xgb",
    model_path: str | None = None,
) -> BaseDiseasePredictor:
    """Factory creating an instantiated BaseDiseasePredictor by registered name."""
    clean_name = name.strip().lower()
    if clean_name not in PREDICTOR_REGISTRY:
        # Import implementations dynamically to register them
        if clean_name in ["baseline_xgb", "xgb", "xgboost"] or clean_name in [
            "calibrated_triad",
            "triad",
            "gbdt_triad",
        ]:
            pass

    if clean_name not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"Predictor '{name}' not found in registry. "
            f"Available predictors: {list(PREDICTOR_REGISTRY.keys())}"
        )

    predictor_cls = PREDICTOR_REGISTRY[clean_name]
    instance = predictor_cls()
    if model_path is not None:
        instance.load(model_path)
    return instance
