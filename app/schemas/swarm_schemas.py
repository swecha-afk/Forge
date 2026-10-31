"""Pydantic v2 schemas for the Agritech Swarm & Risk/RAG subsystems.

Enforces immutable runtime contracts between ML Risk Models, RAG retrieval,
and Agentic LangGraph execution nodes.
"""

from pydantic import BaseModel, Field, field_validator

# Locked schema constants
SUPPORTED_CROPS: list[str] = ["potato", "rice", "wheat", "tomato"]
CROP_TO_ID: dict[str, int] = {
    "potato": 0,
    "rice": 1,
    "wheat": 2,
    "tomato": 3,
}
RISK_THRESHOLD_MODERATE: float = 0.35
RISK_THRESHOLD_HIGH: float = 0.65
RISK_THRESHOLD_SEVERE: float = 0.85


class WeatherTelemetryInput(BaseModel):
    """Raw meteorological and environmental telemetry from Open-Meteo / IMD sensors."""

    crop_name: str = Field(..., description="Crop name (potato, rice, wheat, tomato)")
    temp_min_c: float = Field(..., description="Minimum night temperature (deg C)")
    temp_max_c: float = Field(..., description="Maximum day temperature (deg C)")
    humidity_3d_avg: float = Field(
        ..., ge=0.0, le=100.0, description="3-day rolling relative humidity percentage"
    )
    rain_7d_mm: float = Field(
        ..., ge=0.0, description="7-day cumulative rainfall in mm"
    )
    soil_moisture: float = Field(
        ..., ge=0.0, le=1.0, description="Topsoil volumetric moisture m^3/m^3 (0-7cm)"
    )
    rain_next_24h_mm: float = Field(
        default=0.0, ge=0.0, description="Forecast precipitation in next 24h"
    )
    wind_speed_kmh: float = Field(
        default=0.0, ge=0.0, description="Surface wind speed at 10m in km/h"
    )
    soil_moisture_root_zone: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Root zone volumetric moisture m^3/m^3 (7-28cm)",
    )

    @field_validator("crop_name")
    @classmethod
    def validate_crop(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in SUPPORTED_CROPS:
            raise ValueError(
                f"Crop '{v}' is not supported. Supported crops: {SUPPORTED_CROPS}"
            )
        return clean

    def to_feature_vector(self) -> list[float]:
        """Maps telemetry to the locked 6-dimensional feature vector:

        [temp_min_c, temp_max_c, humidity_3d_avg, rain_7d_mm, soil_moisture, crop_id]
        """
        crop_id = float(CROP_TO_ID[self.crop_name])
        return [
            float(self.temp_min_c),
            float(self.temp_max_c),
            float(self.humidity_3d_avg),
            float(self.rain_7d_mm),
            float(self.soil_moisture),
            crop_id,
        ]


class ShapFeatureDriver(BaseModel):
    """Individual feature attribution from exact TreeSHAP."""

    feature_name: str
    feature_value: float
    shap_value: float
    effect: str = Field(description="'increases_risk' or 'decreases_risk'")


class StandardizedRiskOutput(BaseModel):
    """Immutable contract returned by all BaseDiseasePredictor models."""

    predicted_disease: str
    predicted_disease_id: int
    disease_probability: float = Field(..., ge=0.0, le=1.0)
    calibrated_probability: float = Field(..., ge=0.0, le=1.0)
    base_value_log_odds: float
    shap_top_drivers: list[ShapFeatureDriver]
    raw_shap_vector: list[float] = Field(
        ..., description="Length-6 vector matching FEATURE_ORDER"
    )
    model_version: str
    risk_level: str

    @classmethod
    def compute_risk_level(cls, prob: float) -> str:
        if prob < RISK_THRESHOLD_MODERATE:
            return "LOW"
        if prob < RISK_THRESHOLD_HIGH:
            return "MODERATE"
        if prob < RISK_THRESHOLD_SEVERE:
            return "HIGH"
        return "SEVERE"


class ICARCitation(BaseModel):
    """Verified government agronomic guidance with page-level provenance."""

    chunk_id: int
    source_doc_title: str
    page_number: int
    gov_url: str
    crop_name: str
    disease_or_pest: str
    active_ingredient: str
    dilution_dosage: str
    phi_days: int
    raw_content: str
    score: float


class SafetyOverride(BaseModel):
    """Deterministic cross-node guardrails (Spray Lock & Leaching Gate)."""

    hold_spray: bool = False
    hold_irrigation: bool = False
    override_reasons: list[str] = Field(default_factory=list)
    safe_spray_window_hours: str | None = None
    action_recommendation: str


class RAGQueryResponse(BaseModel):
    """Contract returned by DeterministicICARRAG retriever."""

    status: str = Field(
        description="'SUCCESS', 'LOW_CONFIDENCE_GUARDRAIL_TRIGGERED', 'SUBSPACE_EMPTY'"
    )
    crop_name: str
    disease_or_pest: str
    citations: list[ICARCitation] = Field(default_factory=list)
    top_similarity_score: float = 0.0
    chemical_synthesis_blocked: bool = False
    guardrail_message: str | None = None


class SwarmExecutionResult(BaseModel):
    """Final unified payload passed to FastAPI / Streamlit / Multimodal Voice."""

    crop_name: str
    risk_assessment: StandardizedRiskOutput
    rag_guidance: RAGQueryResponse
    safety_guardrails: SafetyOverride
    executive_advisory: str
    timestamp_utc: str
