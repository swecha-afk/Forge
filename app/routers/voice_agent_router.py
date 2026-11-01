"""FastAPI Router for Groq voice STT and LangGraph workflow orchestration."""

import time
from typing import Any
from fastapi import APIRouter, HTTPException

from app.schemas.voice_schemas import (
    LANGUAGE_CODE_TO_NAME,
    AgentVoiceQueryRequest,
    AgentVoiceQueryResponse,
    VoiceTranscriptionRequest,
    VoiceTranscriptionResponse,
)
from app.services.langgraph_agent.workflow import run_agritech_agent_workflow
from app.services.voice_service.bhashini_stt import (
    BhashiniVoiceService,
    VoiceServiceUnavailableError,
)

router = APIRouter(prefix="/api/v1/agent", tags=["Voice STT & Workflow Orchestration"])


@router.post("/voice/transcribe", response_model=VoiceTranscriptionResponse)
async def transcribe_voice_note(
    payload: VoiceTranscriptionRequest,
) -> VoiceTranscriptionResponse:
    """Transcribes a voice note with Groq and returns vernacular and English text."""
    service = BhashiniVoiceService()
    try:
        return await service.transcribe_and_translate(payload)
    except VoiceServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Voice transcription failed: {str(e)}"
        )


@router.post("/orchestrate", response_model=AgentVoiceQueryResponse)
async def orchestrate_voice_query(
    payload: AgentVoiceQueryRequest,
) -> AgentVoiceQueryResponse:
    """Executes full LangGraph orchestration from voice or text input:

    Node 1: Groq Whisper STT (Vernacular transcription and English translation)
    Node 2: Agronomic Router (Crop, Location, Intent)
    Node 3: Intelligence Swarm (Triad ML / ICAR RAG / Mandi Harvest Advisor)
    Node 4: Deterministic Guardrails (Spray-Lock / Leaching Gate / Spoilage Override)
    Node 5: Vernacular Synthesis (Native Speech Text)
    Node 6: Sarvam TTS output in the selected response language
    """
    t0 = time.perf_counter()
    try:
        final_state = await run_agritech_agent_workflow(
            audio_base64=payload.audio_base64,
            audio_format=payload.audio_format,
            text_query=payload.text_query,
            source_language=payload.source_language,
            district=payload.district,
            state=payload.state,
            crop_override=payload.crop_override,
            latitude=payload.latitude,
            longitude=payload.longitude,
        )
    except Exception as e:
        if isinstance(e, VoiceServiceUnavailableError):
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(
            status_code=500, detail=f"LangGraph Orchestration failed: {str(e)}"
        )

    t1 = time.perf_counter()

    # Extract details payload
    details_payload: dict[str, Any] = {}
    if final_state.get("engine_result"):
        details_payload["agronomic_engine"] = final_state["engine_result"]
    if final_state.get("mandi_result"):
        details_payload["mandi_advisor"] = final_state["mandi_result"]
    if final_state.get("weather_telemetry"):
        details_payload["weather_telemetry"] = final_state["weather_telemetry"]
    if final_state.get("insurance_guidance"):
        details_payload["insurance_guidance"] = final_state["insurance_guidance"]

    return AgentVoiceQueryResponse(
        farmer_vernacular_transcript=final_state.get("farmer_vernacular_transcript", ""),
        english_transcript=final_state.get("english_transcript", ""),
        source_language=final_state.get("source_language", "hi"),
        detected_crop=final_state.get("detected_crop", "potato"),
        intent=final_state.get("intent", "disease_outbreak"),
        intent_classification_source=final_state.get(
            "intent_classification_source", "KEYWORD_FALLBACK"
        ),
        district=final_state.get("district", payload.district),
        state=final_state.get("state", payload.state),
        executive_advisory_en=final_state.get("executive_advisory_en", ""),
        vernacular_response_text=final_state.get("vernacular_response_text", ""),
        safety_guardrails_active=final_state.get("active_guardrails", []),
        execution_route=final_state.get("execution_route", "UNIFIED_AGRONOMIC_ENGINE"),
        data_source_voice=final_state.get("data_source_voice", "DIRECT_TEXT"),
        tts_audio_base64=final_state.get("tts_audio_base64"),
        tts_status=final_state.get("tts_status", "TEXT_ONLY"),
        tts_mime_type=final_state.get("tts_mime_type", "audio/wav"),
        tts_error=final_state.get("tts_error"),
        farmer_response_source=final_state.get(
            "farmer_response_source", "TEMPLATE_FALLBACK"
        ),
        llm_error=final_state.get("llm_error"),
        details=details_payload,
    )


@router.get("/languages", tags=["Voice STT & Architecture Orchestration"])
async def get_supported_languages() -> dict[str, str]:
    """Returns mapping of supported Indic vernacular language codes to official names."""
    return LANGUAGE_CODE_TO_NAME
