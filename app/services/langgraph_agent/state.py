"""LangGraph state schema for Agritech Swarm Architecture Orchestration."""

from typing import Any, Literal, TypedDict


class AgritechAgentState(TypedDict, total=False):
    """Immutable state dictionary flowing through all LangGraph nodes."""

    # 1. Spoken Audio / Input Text
    audio_base64: str | None
    audio_format: str
    source_language: str  # hi, bn, te, mr, en
    farmer_vernacular_transcript: str
    english_transcript: str
    data_source_voice: str
    intent_classification_source: str
    farmer_response_source: str
    llm_error: str | None

    # 2. Extracted Entities & Routing
    detected_crop: str  # potato, tomato, rice, wheat
    district: str
    state: str
    latitude: float
    longitude: float
    intent: Literal[
        "disease_outbreak",
        "harvest_scheduling",
        "irrigation_check",
        "crop_insurance",
        "general",
    ]

    # 3. Tool Execution & Swarm Results
    weather_telemetry: dict[str, Any] | None
    weather_data_source: str
    engine_result: dict[str, Any] | None
    mandi_result: dict[str, Any] | None
    insurance_guidance: dict[str, Any] | None
    general_query_context: dict[str, Any] | None
    execution_route: str

    # 4. Deterministic Guardrails Check
    active_guardrails: list[str]
    spray_lock_active: bool
    irrigation_lock_active: bool
    spoilage_override_active: bool

    # 5. Vernacular Output Synthesis
    executive_advisory_en: str
    vernacular_response_text: str
    final_speech_text: str

    # 6. TTS Audio Output
    tts_audio_base64: str | None
    tts_status: str  # "TTS_LIVE", "TTS_UNAVAILABLE", "TEXT_ONLY"
    tts_mime_type: str
    tts_error: str | None
