"""Baseline XGBoost Disease Risk Predictor with Exact TreeSHAP Attribution.

Provides the v1 prototype for epidemiological risk modeling, returning the
StandardizedRiskOutput contract with game-theoretic feature attribution.
"""

import os

import joblib
import numpy as np
import shap
import xgboost as xgb

from app.schemas.swarm_schemas import (
    ShapFeatureDriver,
    StandardizedRiskOutput,
    WeatherTelemetryInput,
)
from app.services.risk_engine.base import BaseDiseasePredictor, register_predictor
from app.services.risk_engine.constants import (
    DISEASE_TO_ID,
    FEATURE_ORDER,
    ID_TO_DISEASE,
)


@register_predictor("baseline_xgb")
@register_predictor("xgb")
class BaselineXGBPredictor(BaseDiseasePredictor):
    """XGBoost multiclass predictor with exact polynomial-time TreeSHAP attribution."""

    model_name: str = "BaselineXGBPredictor"
    version: str = "1.0.0-xgb"

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.08
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: xgb.XGBClassifier | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineXGBPredictor":
        """Fits XGBoost classifier and builds the TreeSHAP explainer."""
        self.classes_ = np.unique(y)
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(DISEASE_TO_ID),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="mlogloss",
        )
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def predict_and_explain(
        self, telemetry: WeatherTelemetryInput
    ) -> StandardizedRiskOutput:
        """Executes forward inference and extracts exact Shapley values for winning class."""
        if self.model is None or self.explainer is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load() before predicting."
            )

        feat_vector = telemetry.to_feature_vector()
        x_2d = np.array([feat_vector], dtype=np.float32)

        # Predict multi-class probabilities
        probs = self.model.predict_proba(x_2d)[0]
        k_star = int(np.argmax(probs))
        p_k_star = float(probs[k_star])
        disease_name = ID_TO_DISEASE.get(k_star, "unknown")

        # Compute exact TreeSHAP values for input x
        sv = self.explainer.shap_values(x_2d)

        # Handle TreeExplainer return structure across different formats
        if isinstance(sv, list):
            # List of arrays [num_classes], each (num_samples, num_features)
            raw_shap = sv[k_star][0]
            base_val = float(self.explainer.expected_value[k_star])
        elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
            # Array of shape (num_samples, num_features, num_classes)
            raw_shap = sv[0, :, k_star]
            base_val = float(self.explainer.expected_value[k_star])
        else:
            raw_shap = sv[0]
            base_val = 0.0

        raw_shap_list = [float(val) for val in raw_shap]

        # Extract top 3 drivers sorted by absolute Shapley magnitude |phi_i| descending
        indexed_drivers = []
        for i, feat_name in enumerate(FEATURE_ORDER):
            phi_i = raw_shap_list[i]
            x_val = feat_vector[i]
            indexed_drivers.append(
                ShapFeatureDriver(
                    feature_name=feat_name,
                    feature_value=float(x_val),
                    shap_value=round(phi_i, 4),
                    effect="increases_risk" if phi_i > 0 else "decreases_risk",
                )
            )

        top_drivers = sorted(
            indexed_drivers, key=lambda d: abs(d.shap_value), reverse=True
        )[:3]

        risk_level = StandardizedRiskOutput.compute_risk_level(p_k_star)

        return StandardizedRiskOutput(
            predicted_disease=disease_name,
            predicted_disease_id=k_star,
            disease_probability=round(p_k_star, 4),
            calibrated_probability=round(p_k_star, 4),
            base_value_log_odds=round(base_val, 4),
            shap_top_drivers=top_drivers,
            raw_shap_vector=[round(v, 4) for v in raw_shap_list],
            model_version=self.version,
            risk_level=risk_level,
        )

    def save(self, filepath: str) -> None:
        """Persists model and metadata using joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "classes_": self.classes_,
                "version": self.version,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
            filepath,
        )

    def load(self, filepath: str) -> "BaselineXGBPredictor":
        """Loads model from disk and reinitializes TreeExplainer."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")
        data = joblib.load(filepath)
        self.model = data["model"]
        self.classes_ = data["classes_"]
        self.version = data.get("version", self.version)
        self.explainer = shap.TreeExplainer(self.model)
        return self
