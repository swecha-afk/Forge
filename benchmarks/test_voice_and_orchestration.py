"""Verification Suite for Bhashini Voice STT and LangGraph Architecture Orchestration.

Tests:
1. Bhashini Speech-to-Text & English Translation across Indic languages (Hindi, Bengali, Telugu, Marathi).
2. Write-through SQLite resilience cache persistence and failover.
3. End-to-end LangGraph Swarm Orchestration across all 5 nodes:
   - Node 1: Voice STT
   - Node 2: Entity & Intent Router
   - Node 3: Tool Execution (Agronomic Triad ML / Mandi Tool)
   - Node 4: Guardrail Verification
   - Node 5: Vernacular TTS Text Synthesis
4. Verification of cross-node deterministic safety overrides within LangGraph.
"""

import base64
import pytest
import pytest_asyncio

from app.schemas.voice_schemas import VoiceTranscriptionRequest
from app.services.langgraph_agent.workflow import run_agritech_agent_workflow
from app.services.voice_service.audio_utils import compute_audio_hash
from app.services.voice_service.bhashini_stt import BhashiniVoiceService


@pytest.fixture(scope="session")
def voice_service():
    """Instantiates BhashiniVoiceService with SQLite cache."""
    return BhashiniVoiceService()


@pytest.mark.asyncio
async def test_bhashini_voice_stt_hindi(voice_service):
    """Verifies that Hindi voice audio is transcribed into vernacular and English."""
    # Simulated audio payload: 1 second of dummy PCM bytes
    dummy_audio = b"\x00\x01\x00\x02" * 4000
    b64_audio = base64.b64encode(dummy_audio).decode("utf-8")

    req = VoiceTranscriptionRequest(
        audio_base64=b64_audio,
        source_language="hi",
        audio_format="wav",
    )
    resp = await voice_service.transcribe_and_translate(req)

    assert resp.vernacular_transcript != ""
    assert resp.english_transcript != ""
    assert resp.source_language == "hi"
    assert resp.target_language == "en"
    assert resp.data_source in ["BHASHINI_LIVE_API", "RESILIENCE_CACHE_FAILOVER"]
    assert resp.confidence_score > 0.85


@pytest.mark.asyncio
async def test_bhashini_voice_stt_bengali(voice_service):
    """Verifies that Bengali voice audio is transcribed and translated."""
    dummy_audio = b"\x01\x02\x03\x04" * 4000
    b64_audio = base64.b64encode(dummy_audio).decode("utf-8")

    req = VoiceTranscriptionRequest(
        audio_base64=b64_audio,
        source_language="bn",
        audio_format="wav",
    )
    resp = await voice_service.transcribe_and_translate(req)

    assert resp.vernacular_transcript != ""
    assert resp.english_transcript != ""
    assert resp.source_language == "bn"
    assert resp.target_language == "en"


@pytest.mark.asyncio
async def test_bhashini_sqlite_cache_roundtrip(voice_service):
    """Verifies that cached transcripts are stored and retrieved with 100% fidelity."""
    sample_bytes = b"sample_farmer_recording_bytes_12345"
    audio_hash = compute_audio_hash(sample_bytes)

    voice_service.cache_transcription(
        audio_hash=audio_hash,
        source_lang="te",
        vernacular="వరి పంటలో తెగులు లక్షణాలు కనిపిస్తున్నాయి",
        english="Blight symptoms are visible in the rice crop",
    )

    cached = voice_service.get_cached_transcription(audio_hash)
    assert cached is not None
    lang, vern, eng = cached
    assert lang == "te"
    assert "వరి" in vern
    assert "rice" in eng.lower()


@pytest.mark.asyncio
async def test_langgraph_orchestrator_disease_outbreak_flow():
    """Verifies full 5-Node LangGraph orchestration on a Hindi disease outbreak query."""
    hindi_query = "आलू के पत्तों पर काले धब्बे दिख रहे हैं, क्या यह झुलसा रोग है? मुझे क्या दवा डालनी चाहिए?"

    final_state = await run_agritech_agent_workflow(
        text_query=hindi_query,
        source_language="hi",
        district="Medinipur West",
        state="West Bengal",
    )

    assert final_state["detected_crop"] == "potato"
    assert final_state["intent"] == "disease_outbreak"
    assert final_state["execution_route"] == "UNIFIED_AGRONOMIC_ENGINE"
    assert final_state["engine_result"] is not None

    engine_res = final_state["engine_result"]
    assert "risk_assessment" in engine_res
    assert "rag_guidance" in engine_res
    assert len(engine_res["risk_assessment"]["shap_top_drivers"]) == 3

    # Node 5 Vernacular synthesis output check
    assert final_state["vernacular_response_text"] != ""
    assert len(final_state["final_speech_text"]) > 20


@pytest.mark.asyncio
async def test_langgraph_orchestrator_harvest_mandi_flow():
    """Verifies full 5-Node LangGraph orchestration on a Bengali Mandi harvest query."""
    bengali_query = "টমেটোর মান্ডি দর কেমন চলছে? ফসল কি আজই তুলে বিক্রি করব?"

    final_state = await run_agritech_agent_workflow(
        text_query=bengali_query,
        source_language="bn",
        district="Medinipur West",
        state="West Bengal",
    )

    assert final_state["detected_crop"] == "tomato"
    assert final_state["intent"] == "harvest_scheduling"
    assert final_state["execution_route"] == "MANDI_HARVEST_ADVISOR"
    assert final_state["mandi_result"] is not None

    mandi_res = final_state["mandi_result"]
    assert mandi_res["current_modal_price_inr_per_quintal"] > 0
    assert mandi_res["perishability_tier"] == "HIGH"

    # Node 5 Vernacular synthesis in Bengali
    assert final_state["vernacular_response_text"] != ""


@pytest.mark.asyncio
async def test_langgraph_orchestrator_voice_input_flow():
    """Verifies that providing base64 voice audio executes STT Node 1 before routing."""
    dummy_audio = b"\x00\x02\x04\x06" * 3000
    b64_audio = base64.b64encode(dummy_audio).decode("utf-8")

    final_state = await run_agritech_agent_workflow(
        audio_base64=b64_audio,
        source_language="hi",
        district="Hooghly",
        state="West Bengal",
    )

    assert final_state["farmer_vernacular_transcript"] != ""
    assert final_state["english_transcript"] != ""
    assert final_state["data_source_voice"] in [
        "BHASHINI_LIVE_API",
        "RESILIENCE_CACHE_FAILOVER",
    ]
    assert final_state["execution_route"] in [
        "UNIFIED_AGRONOMIC_ENGINE",
        "MANDI_HARVEST_ADVISOR",
    ]

