"""FastAPI Router for Agronomic Intelligence, Risk Inference, and Subspace RAG."""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.swarm_schemas import (
    RAGQueryResponse,
    StandardizedRiskOutput,
    SwarmExecutionResult,
    WeatherTelemetryInput,
)
from app.services.rag_engine.retriever import DeterministicICARRAG
from app.services.risk_engine.base import get_predictor
from app.services.risk_engine.pipeline import UnifiedAgronomicEngine
from app.services.weather_api import LiveWeatherService

router = APIRouter(prefix="/api/v1", tags=["Agronomic Intelligence & RAG Engine"])

# Initialize singleton engines (lazy-loaded on startup)
_triad_predictor = None
_rag_retriever = None
_unified_engine = None
_weather_service = None


def get_engine() -> UnifiedAgronomicEngine:
    global _triad_predictor, _rag_retriever, _unified_engine
    if _unified_engine is None:
        try:
            _triad_predictor = get_predictor(
                "calibrated_triad", "artifacts/calibrated_triad.joblib"
            )
        except Exception:
            _triad_predictor = get_predictor(
                "baseline_xgb", "artifacts/xgb_baseline.joblib"
            )
        _rag_retriever = DeterministicICARRAG()
        _unified_engine = UnifiedAgronomicEngine(
            predictor=_triad_predictor, rag_retriever=_rag_retriever
        )
    return _unified_engine


def get_weather_service() -> LiveWeatherService:
    global _weather_service
    if _weather_service is None:
        _weather_service = LiveWeatherService()
    return _weather_service


@router.post("/advisory", response_model=SwarmExecutionResult)
async def generate_agronomic_advisory(
    telemetry: WeatherTelemetryInput,
    farmer_query: str = Query(
        default="What is the recommended treatment and safety protocol?",
        description="Farmer's specific question or voice prompt",
    ),
) -> SwarmExecutionResult:
    """Executes the complete Agronomic Swarm Intelligence pipeline:

    Weather Telemetry -> Calibrated Risk Prediction -> Exact TreeSHAP ->
    ICAR Subspace RAG -> Deterministic Safety Overrides (Spray Lock & Leaching Gate).
    """
    engine = get_engine()
    return engine.execute(telemetry=telemetry, farmer_query=farmer_query)


@router.post("/weather/live", response_model=WeatherTelemetryInput)
async def get_live_weather_telemetry(
    crop_name: str = Query(..., description="Crop name: potato, rice, wheat, tomato"),
    latitude: float = Query(..., description="Latitude coordinate"),
    longitude: float = Query(..., description="Longitude coordinate"),
) -> WeatherTelemetryInput:
    """Fetches real-time Open-Meteo telemetry for any Indian latitude/longitude with SQLite failover."""
    service = get_weather_service()
    try:
        return service.fetch_live_telemetry(
            crop_name=crop_name, latitude=latitude, longitude=longitude
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/predict-risk", response_model=StandardizedRiskOutput)
async def predict_disease_risk(
    telemetry: WeatherTelemetryInput,
    model_family: str = Query(
        default="calibrated_triad", description="'calibrated_triad' or 'baseline_xgb'"
    ),
) -> StandardizedRiskOutput:
    """Runs forward inference on the selected model family and returns exact TreeSHAP feature attributions."""
    try:
        if model_family.lower() in ["xgb", "baseline_xgb"]:
            predictor = get_predictor("baseline_xgb", "artifacts/xgb_baseline.joblib")
        else:
            predictor = get_predictor(
                "calibrated_triad", "artifacts/calibrated_triad.joblib"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load model {model_family}: {e}"
        )

    return predictor.predict_and_explain(telemetry)


@router.post("/rag/retrieve", response_model=RAGQueryResponse)
async def retrieve_icar_guidance(
    crop_name: str = Query(..., description="Crop name"),
    disease_or_pest: str = Query(..., description="Diagnosed condition or pest"),
    farmer_query: str = Query(
        ..., description="Specific question regarding dilution, dosage, or timing"
    ),
    top_k: int = Query(default=3, ge=1, le=5),
) -> RAGQueryResponse:
    """Executes subspace-isolated ICAR guidance retrieval with RRF and OOD Confidence Gate."""
    engine = get_engine()
    return engine.rag.retrieve(
        crop_name=crop_name,
        disease_or_pest=disease_or_pest,
        farmer_query=farmer_query,
        top_k=top_k,
    )
