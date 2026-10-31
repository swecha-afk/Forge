"""Calibrated GBDT Triad Predictor with Exact Linear Ensemble TreeSHAP.

Ensemble of XGBoost + LightGBM + CatBoost with Isotonic Probability Calibration
(CalibratedClassifierCV) and linear game-theoretic Shapley value aggregation:
phi_ens(x) = sum(w_m * phi_m(x)).
"""

import os

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

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


@register_predictor("calibrated_triad")
@register_predictor("triad")
@register_predictor("gbdt_triad")
class CalibratedTriadPredictor(BaseDiseasePredictor):
    """Ensemble of XGBoost, LightGBM, and CatBoost with Isotonic Probability Calibration."""

    model_name: str = "CalibratedTriadPredictor"
    version: str = "2.0.0-triad-calibrated"

    def __init__(self, weights: list[float] | None = None):
        # Default weights: 35% XGBoost, 35% LightGBM, 30% CatBoost
        self.weights = weights or [0.35, 0.35, 0.30]
        assert len(self.weights) == 3, "Triad requires exactly 3 model weights"
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

        self.calibrated_xgb: CalibratedClassifierCV | None = None
        self.calibrated_lgb: CalibratedClassifierCV | None = None
        self.calibrated_cat: CalibratedClassifierCV | None = None

        self.base_xgb: xgb.XGBClassifier | None = None
        self.base_lgb: lgb.LGBMClassifier | None = None
        self.base_cat: cb.CatBoostClassifier | None = None

        self.explainer_xgb: shap.TreeExplainer | None = None
        self.explainer_lgb: shap.TreeExplainer | None = None
        self.explainer_cat: shap.TreeExplainer | None = None

        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedTriadPredictor":
        """Fits base GBDT triad models and their isotonic probability calibrators."""
        self.classes_ = np.unique(y)
        num_classes = len(DISEASE_TO_ID)

        # 1. Base Tree Estimators (for exact TreeSHAP attribution)
        self.base_xgb = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="mlogloss",
        )
        self.base_xgb.fit(X, y)
        self.explainer_xgb = shap.TreeExplainer(self.base_xgb)

        self.base_lgb = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
        )
        self.base_lgb.fit(X, y)
        self.explainer_lgb = shap.TreeExplainer(self.base_lgb)

        self.base_cat = cb.CatBoostClassifier(
            loss_function="MultiClass",
            iterations=100,
            depth=4,
            learning_rate=0.08,
            random_seed=42,
            verbose=0,
        )
        self.base_cat.fit(X, y)
        self.explainer_cat = shap.TreeExplainer(self.base_cat)

        # 2. Isotonically Calibrated Classifiers (for reliable probability estimates)
        self.calibrated_xgb = CalibratedClassifierCV(
            estimator=xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                eval_metric="mlogloss",
            ),
            method="isotonic",
            cv=3,
        )
        self.calibrated_xgb.fit(X, y)

        self.calibrated_lgb = CalibratedClassifierCV(
            estimator=lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                verbose=-1,
            ),
            method="isotonic",
            cv=3,
        )
        self.calibrated_lgb.fit(X, y)

        self.calibrated_cat = CalibratedClassifierCV(
            estimator=cb.CatBoostClassifier(
                loss_function="MultiClass",
                iterations=80,
                depth=4,
                learning_rate=0.08,
                random_seed=42,
                verbose=0,
            ),
            method="isotonic",
            cv=3,
        )
        self.calibrated_cat.fit(X, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns weighted ensemble calibrated probabilities: P_ens = sum(w_m * P_m)."""
        if self.calibrated_xgb is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load() before predicting."
            )

        p_xgb = self.calibrated_xgb.predict_proba(X)
        p_lgb = self.calibrated_lgb.predict_proba(X)
        p_cat = self.calibrated_cat.predict_proba(X)

        w1, w2, w3 = self.weights
        p_ens = (w1 * p_xgb) + (w2 * p_lgb) + (w3 * p_cat)
        # Re-normalize across classes
        p_ens = p_ens / np.sum(p_ens, axis=1, keepdims=True)
        return p_ens

    def predict_and_explain(
        self, telemetry: WeatherTelemetryInput
    ) -> StandardizedRiskOutput:
        """Runs calibrated ensemble inference and linearly aggregates TreeSHAP attributions."""
        feat_vector = telemetry.to_feature_vector()
        x_2d = np.array([feat_vector], dtype=np.float32)

        # 1. Calibrated Ensemble Probabilities
        calibrated_probs = self.predict_proba(x_2d)[0]
        k_star = int(np.argmax(calibrated_probs))
        p_calibrated = float(calibrated_probs[k_star])
        disease_name = ID_TO_DISEASE.get(k_star, "unknown")

        # Raw (uncalibrated) base ensemble probability for comparison
        raw_p_xgb = self.base_xgb.predict_proba(x_2d)[0][k_star]
        raw_p_lgb = self.base_lgb.predict_proba(x_2d)[0][k_star]
        raw_p_cat = self.base_cat.predict_proba(x_2d)[0][k_star]
        w1, w2, w3 = self.weights
        raw_prob = float(w1 * raw_p_xgb + w2 * raw_p_lgb + w3 * raw_p_cat)

        # 2. Linear Ensemble TreeSHAP: phi_ens = sum(w_m * phi_m)
        def _extract_shap(sv_result, explainer, class_idx):
            if isinstance(sv_result, np.ndarray) and len(sv_result.shape) == 3:
                vec = sv_result[0, :, class_idx]
            elif isinstance(sv_result, list):
                vec = sv_result[class_idx][0]
            else:
                vec = sv_result[0]
            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                base = float(exp_val[class_idx])
            else:
                base = float(exp_val)
            return vec, base

        sv_xgb = self.explainer_xgb.shap_values(x_2d)
        sv_lgb = self.explainer_lgb.shap_values(x_2d)
        sv_cat = self.explainer_cat.shap_values(x_2d)

        phi_xgb, b_xgb = _extract_shap(sv_xgb, self.explainer_xgb, k_star)
        phi_lgb, b_lgb = _extract_shap(sv_lgb, self.explainer_lgb, k_star)
        phi_cat, b_cat = _extract_shap(sv_cat, self.explainer_cat, k_star)

        phi_ens = (w1 * phi_xgb) + (w2 * phi_lgb) + (w3 * phi_cat)
        base_log_odds = (w1 * b_xgb) + (w2 * b_lgb) + (w3 * b_cat)

        raw_shap_list = [float(val) for val in phi_ens]

        # 3. Top 3 drivers sorted by absolute Shapley magnitude |phi_i| descending
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

        risk_level = StandardizedRiskOutput.compute_risk_level(p_calibrated)

        return StandardizedRiskOutput(
            predicted_disease=disease_name,
            predicted_disease_id=k_star,
            disease_probability=round(raw_prob, 4),
            calibrated_probability=round(p_calibrated, 4),
            base_value_log_odds=round(float(base_log_odds), 4),
            shap_top_drivers=top_drivers,
            raw_shap_vector=[round(v, 4) for v in raw_shap_list],
            model_version=self.version,
            risk_level=risk_level,
        )

    def save(self, filepath: str) -> None:
        """Persists triad models and calibration wrappers."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(
            {
                "calibrated_xgb": self.calibrated_xgb,
                "calibrated_lgb": self.calibrated_lgb,
                "calibrated_cat": self.calibrated_cat,
                "base_xgb": self.base_xgb,
                "base_lgb": self.base_lgb,
                "base_cat": self.base_cat,
                "weights": self.weights,
                "classes_": self.classes_,
                "version": self.version,
            },
            filepath,
        )

    def load(self, filepath: str) -> "CalibratedTriadPredictor":
        """Loads triad models from disk and reinitializes TreeExplainers."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Triad checkpoint not found: {filepath}")
        data = joblib.load(filepath)
        self.calibrated_xgb = data["calibrated_xgb"]
        self.calibrated_lgb = data["calibrated_lgb"]
        self.calibrated_cat = data["calibrated_cat"]
        self.base_xgb = data["base_xgb"]
        self.base_lgb = data["base_lgb"]
        self.base_cat = data["base_cat"]
        self.weights = data.get("weights", self.weights)
        self.classes_ = data.get("classes_")
        self.version = data.get("version", self.version)

        self.explainer_xgb = shap.TreeExplainer(self.base_xgb)
        self.explainer_lgb = shap.TreeExplainer(self.base_lgb)
        self.explainer_cat = shap.TreeExplainer(self.base_cat)
        return self
