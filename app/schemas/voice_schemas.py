"""Pydantic v2 schemas for voice I/O, multilingual translation, and workflow contracts."""

from typing import Any, Literal
from pydantic import BaseModel, Field

SupportedLanguage = Literal["hi", "bn", "te", "mr", "ta", "kn", "pa", "gu", "en"]

# Mapping vernacular language codes to official display names
LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "kn": "Kannada",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "en": "English",
}


class VoiceTranscriptionRequest(BaseModel):
    """Input payload containing audio for multilingual speech transcription."""

    audio_base64: str = Field(
        ...,
        description="Base64 encoded audio string (WAV, MP3, OGG, WebM)",
    )
    audio_format: str = Field(
        default="wav",
        description="Audio container format: wav, mp3, ogg, webm, flac",
    )
    source_language: SupportedLanguage = Field(
        default="hi",
        description="Source spoken language code (e.g., 'hi' for Hindi, 'bn' for Bengali, 'te' for Telugu, 'mr' for Marathi)",
    )
    sampling_rate: int = Field(
        default=16000,
        description="Audio sampling rate in Hz (standard: 16000 or 8000)",
    )


class VoiceTranscriptionResponse(BaseModel):
    """Output contract for Speech-to-Text & English Translation."""

    vernacular_transcript: str = Field(
        ..., description="Raw transcript in original spoken Indic script"
    )
    english_transcript: str = Field(
        ..., description="Standardized English translation of the spoken transcript"
    )
    source_language: str = Field(
        ..., description="Detected or specified source language code"
    )
    target_language: str = Field(
        default="en", description="Target translated language code"
    )
    data_source: Literal["GROQ_LIVE_API", "RESILIENCE_CACHE_FAILOVER"] = Field(
        ..., description="Data provenance: live Groq API or exact-audio SQLite cache"
    )
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ASR confidence when provided by the transcription provider",
    )
    pipeline_task: str = Field(
        default="groq_whisper_transcription_and_translation",
        description="Speech recognition and translation tasks executed",
    )


class AgentVoiceQueryRequest(BaseModel):
    """Input contract to run end-to-end LangGraph Swarm Orchestration from voice or text."""

    audio_base64: str | None = Field(
        default=None,
        description="Base64 voice note (if audio input). If provided, Groq STT executes first.",
    )
    audio_format: str = Field(
        default="wav",
        description="Format of audio: wav, mp3, ogg, webm",
    )
    text_query: str | None = Field(
        default=None,
        description="Alternative direct text prompt (if audio is omitted)",
    )
    source_language: SupportedLanguage = Field(
        default="hi",
        description="Spoken or written language code (hi, bn, te, mr, en)",
    )
    district: str = Field(
        default="Hooghly",
        description="Farmer's district for weather and APMC mandi rates",
    )
    state: str = Field(
        default="West Bengal",
        description="Farmer's state (e.g., West Bengal, Bihar, Punjab, Karnataka)",
    )
    crop_override: str | None = Field(
        default=None,
        description="Explicit crop name override if known (potato, tomato, rice, wheat)",
    )
    latitude: float | None = Field(
        default=None,
        description="Optional GPS latitude; if omitted, agro-climatic registry coordinates are used",
    )
    longitude: float | None = Field(
        default=None,
        description="Optional GPS longitude; if omitted, agro-climatic registry coordinates are used",
    )


class AgentVoiceQueryResponse(BaseModel):
    """End-to-End Orchestrated Swarm Response delivered to UI / Voice Speaker."""

    farmer_vernacular_transcript: str = Field(
        description="What the farmer said in their native language"
    )
    english_transcript: str = Field(
        description="Standardized English translation of the farmer's query"
    )
    source_language: str
    detected_crop: str
    intent: str = Field(
        description="Detected intent: 'disease_outbreak', 'harvest_scheduling', 'irrigation_check', 'crop_insurance', or 'general'"
    )
    intent_classification_source: str = Field(
        default="KEYWORD_FALLBACK",
        description="Intent source: Groq LLM classification or deterministic fallback",
    )
    district: str
    state: str
    executive_advisory_en: str = Field(
        description="Comprehensive English advisory with ICAR citations, SHAP drivers, and locks"
    )
    vernacular_response_text: str = Field(
        description="Empathetic, actionable vernacular speech text for TTS synthesis in farmer's language"
    )
    safety_guardrails_active: list[str] = Field(
        default_factory=list,
        description="Active safety locks: Spray Lock, Leaching Gate, Mandi Spoilage Override",
    )
    execution_route: str = Field(
        description="Subsystem executed: agronomic engine, mandi advisor, crop-insurance guidance, or general guidance"
    )
    data_source_voice: str
    tts_audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded TTS audio in the farmer's vernacular language (None if TTS unavailable)",
    )
    tts_status: str = Field(
        default="TEXT_ONLY",
        description="TTS output status: 'TTS_LIVE', 'TTS_UNAVAILABLE', or 'TEXT_ONLY'",
    )
    tts_mime_type: str = Field(
        default="audio/wav",
        description="MIME type of the TTS audio output",
    )
    tts_error: str | None = Field(
        default=None,
        description="Reason audio synthesis was unavailable, if applicable",
    )
    farmer_response_source: str = Field(
        default="TEMPLATE_FALLBACK",
        description="Farmer response source: Groq grounded generation or local template",
    )
    llm_error: str | None = Field(
        default=None,
        description="Reason Groq classification or response generation was unavailable",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw JSON execution payloads from risk engine, RAG, or Mandi tool",
    )
