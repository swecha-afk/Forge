"""Nodes for LangGraph Agritech Swarm Orchestrator.

Implements the 5-node agentic workflow:
Node 1: Audio / Speech Input & Bhashini STT
Node 2: Agronomic Entity & Intent Router
Node 3: Intelligence Swarm & Tool Dispatcher
Node 4: Cross-Node Deterministic Guardrail Verification
Node 5: Vernacular Response Synthesis & TTS Output
"""

import logging
from typing import Any

from app.schemas.swarm_schemas import SUPPORTED_CROPS, WeatherTelemetryInput
from app.schemas.voice_schemas import VoiceTranscriptionRequest
from app.services.langgraph_agent.state import AgritechAgentState
from app.services.mandi_service.fetcher import normalize_commodity
from app.services.mandi_service.tool import mandi_harvest_advisor_tool
from app.services.rag_engine.retriever import DeterministicICARRAG
from app.services.risk_engine.base import get_predictor
from app.services.risk_engine.pipeline import UnifiedAgronomicEngine
from app.services.voice_service.bhashini_stt import BhashiniVoiceService
from app.services.weather_api import INDIAN_AGRO_REGIONS, LiveWeatherService

logger = logging.getLogger(__name__)

# Representative Coordinates by Crop / District
DEFAULT_COORDS: dict[str, tuple[float, float]] = {
    "potato": (25.5941, 85.1376),  # Gangetic Potato Belt (Patna / West Bengal)
    "rice": (20.4625, 85.8828),  # NRRI Cuttack Rice Belt
    "wheat": (29.6857, 76.9905),  # IIWBR Karnal NWPZ
    "tomato": (12.9716, 77.5946),  # IIHR Bangalore Tomato Belt
}

DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "medinipur west": (22.4257, 87.3199),
    "hooghly": (22.8963, 88.2461),
    "bankura": (23.2324, 87.0784),
    "patna": (25.5941, 85.1376),
    "karnal": (29.6857, 76.9905),
    "cuttack": (20.4625, 85.8828),
    "bengaluru": (12.9716, 77.5946),
    "pune": (18.5204, 73.8567),
}


# ------------------------------------------------------------------------------
# NODE 1: Audio / Speech Input & Bhashini STT
# ------------------------------------------------------------------------------
async def stt_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 1: Converts voice audio into vernacular transcript and English translation."""
    audio_b64 = state.get("audio_base64")
    src_lang = state.get("source_language", "hi")
    audio_fmt = state.get("audio_format", "wav")

    if audio_b64 and len(audio_b64.strip()) > 10:
        voice_service = BhashiniVoiceService()
        trans_req = VoiceTranscriptionRequest(
            audio_base64=audio_b64,
            audio_format=audio_fmt,
            source_language=src_lang,
        )
        resp = await voice_service.transcribe_and_translate(trans_req)
        return {
            "farmer_vernacular_transcript": resp.vernacular_transcript,
            "english_transcript": resp.english_transcript,
            "data_source_voice": resp.data_source,
        }

    # If text query provided directly
    existing_en = state.get("english_transcript") or state.get(
        "farmer_vernacular_transcript", ""
    )
    return {
        "farmer_vernacular_transcript": existing_en,
        "english_transcript": existing_en,
        "data_source_voice": "DIRECT_TEXT_INPUT",
    }


# ------------------------------------------------------------------------------
# NODE 2: Agronomic Entity & Intent Router
# ------------------------------------------------------------------------------
def router_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 2: Extracts crop entity, geographic coordinates, and farmer operational intent."""
    query = (state.get("english_transcript") or "").lower()
    crop = (state.get("detected_crop") or "").strip().lower()

    # 1. Detect Crop from query if not already provided
    if not crop or crop not in SUPPORTED_CROPS:
        if any(w in query for w in ["potato", "aloo", "alu", "tuber"]):
            crop = "potato"
        elif any(w in query for w in ["tomato", "tamatar", "solanum"]):
            crop = "tomato"
        elif any(w in query for w in ["rice", "paddy", "dhan", "chawal"]):
            crop = "rice"
        elif any(w in query for w in ["wheat", "gehu", "kanak"]):
            crop = "wheat"
        else:
            crop = "potato"  # Default fallback crop

    # 2. Detect Intent
    harvest_keywords = [
        "mandi",
        "price",
        "rate",
        "sell",
        "harvest",
        "market",
        "bhav",
        "daam",
        "spoil",
    ]
    irrigation_keywords = [
        "water",
        "irrigate",
        "irrigation",
        "soil",
        "moisture",
        "urea",
        "fertilizer",
        "pump",
        "hypoxia",
    ]

    if any(kw in query for kw in harvest_keywords):
        intent = "harvest_scheduling"
    elif any(kw in query for kw in irrigation_keywords):
        intent = "irrigation_check"
    else:
        intent = "disease_outbreak"

    # 3. Resolve Lat/Lon
    district = state.get("district", "Medinipur West").strip().lower()
    lat = state.get("latitude")
    lon = state.get("longitude")

    if lat is None or lon is None:
        if district in DISTRICT_COORDS:
            lat, lon = DISTRICT_COORDS[district]
        else:
            lat, lon = DEFAULT_COORDS.get(crop, (25.5941, 85.1376))

    return {
        "detected_crop": crop,
        "intent": intent,
        "latitude": lat,
        "longitude": lon,
    }


# ------------------------------------------------------------------------------
# NODE 3: Intelligence Swarm & Tool Dispatcher
# ------------------------------------------------------------------------------
async def tool_execution_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 3: Dispatches execution to UnifiedAgronomicEngine or MandiHarvestAdvisor."""
    intent = state.get("intent", "disease_outbreak")
    crop = state.get("detected_crop", "potato")
    district = state.get("district", "Medinipur West")
    state_name = state.get("state", "West Bengal")
    lat = state.get("latitude", 25.5941)
    lon = state.get("longitude", 85.1376)
    user_query = (
        state.get("english_transcript")
        or "What is the recommended treatment and safety protocol?"
    )

    if intent == "harvest_scheduling":
        # Fetch upcoming 48h rain from weather API for Spoilage Override check
        weather_svc = LiveWeatherService()
        try:
            weather_data = weather_svc.fetch_live_telemetry(
                crop_name=crop, latitude=lat, longitude=lon
            )
            rain_48h = weather_data.rain_next_24h_mm * 1.5
        except Exception:
            rain_48h = 0.0

        mandi_res = await mandi_harvest_advisor_tool(
            commodity=crop,
            district=district,
            state=state_name,
            rain_forecast_48h_mm=rain_48h,
        )
        return {
            "mandi_result": mandi_res,
            "engine_result": None,
            "execution_route": "MANDI_HARVEST_ADVISOR",
        }

    # Otherwise execute UnifiedAgronomicEngine (ML Risk + TreeSHAP + Subspace ICAR RAG)
    weather_svc = LiveWeatherService()
    weather_telemetry: WeatherTelemetryInput = weather_svc.fetch_live_telemetry(
        crop_name=crop, latitude=lat, longitude=lon
    )

    # Initialize production models
    try:
        model = get_predictor("calibrated_triad", "artifacts/calibrated_triad.joblib")
    except Exception:
        model = get_predictor("baseline_xgb", "artifacts/xgb_baseline.joblib")

    rag = DeterministicICARRAG()
    engine = UnifiedAgronomicEngine(predictor=model, rag_retriever=rag)
    result = engine.execute(telemetry=weather_telemetry, farmer_query=user_query)

    return {
        "weather_telemetry": weather_telemetry.model_dump(),
        "engine_result": result.model_dump(),
        "mandi_result": None,
        "execution_route": "UNIFIED_AGRONOMIC_ENGINE",
    }


# ------------------------------------------------------------------------------
# NODE 4: Cross-Node Deterministic Guardrail Verification
# ------------------------------------------------------------------------------
def guardrail_verification_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 4: Evaluates active cross-node safety locks and flags."""
    active_guardrails = []
    spray_lock = False
    irrigation_lock = False
    spoilage_override = False

    engine_res = state.get("engine_result")
    mandi_res = state.get("mandi_result")

    if engine_res:
        guardrails = engine_res.get("safety_guardrails", {})
        if guardrails.get("hold_spray"):
            spray_lock = True
            active_guardrails.append("[CRITICAL_HOLD_SPRAY] Rain/Wind Lock Active")
        if guardrails.get("hold_irrigation"):
            irrigation_lock = True
            active_guardrails.append("[CRITICAL_HOLD_IRRIGATION] Soil Saturated")
        if engine_res.get("rag_guidance", {}).get("chemical_synthesis_blocked"):
            active_guardrails.append("[OOD_CONFIDENCE_GATE] Chemical Synthesis Blocked")

    if mandi_res:
        if mandi_res.get("action_flag") == "URGENT_HARVEST_WEATHER_RISK":
            spoilage_override = True
            active_guardrails.append(
                "[SPOILAGE_OVERRIDE] Perishable Crop Rain Emergency"
            )

    return {
        "active_guardrails": active_guardrails,
        "spray_lock_active": spray_lock,
        "irrigation_lock_active": irrigation_lock,
        "spoilage_override_active": spoilage_override,
    }


# ------------------------------------------------------------------------------
# NODE 5: Vernacular Response Synthesis & TTS Output
# ------------------------------------------------------------------------------
def synthesizer_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 5: Formulates English summary and native vernacular speech text for TTS audio playback."""
    src_lang = state.get("source_language", "hi")
    crop = state.get("detected_crop", "potato").title()
    voice_service = BhashiniVoiceService()

    engine_res = state.get("engine_result")
    mandi_res = state.get("mandi_result")

    if mandi_res:
        exec_en = mandi_res.get("reasoning_summary", "")
        if mandi_res.get("action_flag") == "URGENT_HARVEST_WEATHER_RISK":
            vern_text = voice_service.synthesize_vernacular_speech_text(
                src_lang, "mandi_spoilage", {}
            )
        else:
            vern_text = voice_service.synthesize_vernacular_speech_text(
                src_lang,
                "mandi_harvest",
                {
                    "market": mandi_res.get("market_name", "Local APMC"),
                    "crop": crop,
                    "price": mandi_res.get("current_modal_price_inr_per_quintal", 0.0),
                    "trend": mandi_res.get("trend_delta_pct", 0.0),
                },
            )
        return {
            "executive_advisory_en": exec_en,
            "vernacular_response_text": vern_text,
            "final_speech_text": vern_text,
        }

    # Otherwise synthesize from UnifiedAgronomicEngine result
    exec_en = engine_res.get("executive_advisory", "") if engine_res else ""
    guardrails = (
        engine_res.get("safety_guardrails", {}) if engine_res else {}
    )
    risk = engine_res.get("risk_assessment", {}) if engine_res else {}
    rag = engine_res.get("rag_guidance", {}) if engine_res else {}
    telemetry = state.get("weather_telemetry", {})

    if guardrails.get("hold_spray"):
        vern_text = voice_service.synthesize_vernacular_speech_text(
            src_lang,
            "spray_lock",
            {
                "rain": telemetry.get("rain_next_24h_mm", 12.0),
                "wind": telemetry.get("wind_speed_kmh", 20.0),
            },
        )
    elif guardrails.get("hold_irrigation"):
        vern_text = voice_service.synthesize_vernacular_speech_text(
            src_lang,
            "irrigation_lock",
            {
                "sm": int(telemetry.get("soil_moisture_root_zone", 0.35) * 100),
            },
        )
    else:
        # Normal scientific advice
        cits = rag.get("citations", [])
        top_cit = cits[0] if cits else {}
        vern_text = voice_service.synthesize_vernacular_speech_text(
            src_lang,
            "disease_advice",
            {
                "crop": crop,
                "disease": risk.get("predicted_disease", "Health Issue").replace(
                    "_", " "
                ),
                "prob": int(risk.get("calibrated_probability", 0.8) * 100),
                "chemical": top_cit.get(
                    "active_ingredient", "अनुशंसित फफूंदनाशक"
                ),
                "dose": top_cit.get("dilution_dosage", "2.5 g/L"),
                "phi": top_cit.get("phi_days", 7),
            },
        )

    return {
        "executive_advisory_en": exec_en,
        "vernacular_response_text": vern_text,
        "final_speech_text": vern_text,
    }

