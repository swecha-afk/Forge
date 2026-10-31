"""Public exports for risk_engine."""

from app.services.risk_engine.base import (
    PREDICTOR_REGISTRY,
    BaseDiseasePredictor,
    get_predictor,
    register_predictor,
)
from app.services.risk_engine.calibrated_triad import CalibratedTriadPredictor
from app.services.risk_engine.constants import (
    CONFIDENCE_GATE_TAU,
    CROP_TO_ID,
    DISEASE_TO_ID,
    FEATURE_ORDER,
    ID_TO_CROP,
    ID_TO_DISEASE,
    RRF_K,
    SOIL_MOISTURE_LEACHING_THRESHOLD,
    SPRAY_RAIN_THRESHOLD_MM,
    SPRAY_WIND_THRESHOLD_KMH,
    SUPPORTED_CROPS,
    SUPPORTED_DISEASES,
)
from app.services.risk_engine.pipeline import UnifiedAgronomicEngine
from app.services.risk_engine.xgb_baseline import BaselineXGBPredictor

__all__ = [
    "CONFIDENCE_GATE_TAU",
    "CROP_TO_ID",
    "DISEASE_TO_ID",
    "FEATURE_ORDER",
    "ID_TO_CROP",
    "ID_TO_DISEASE",
    "PREDICTOR_REGISTRY",
    "RRF_K",
    "SOIL_MOISTURE_LEACHING_THRESHOLD",
    "SPRAY_RAIN_THRESHOLD_MM",
    "SPRAY_WIND_THRESHOLD_KMH",
    "SUPPORTED_CROPS",
    "SUPPORTED_DISEASES",
    "BaseDiseasePredictor",
    "BaselineXGBPredictor",
    "CalibratedTriadPredictor",
    "UnifiedAgronomicEngine",
    "get_predictor",
    "register_predictor",
]
