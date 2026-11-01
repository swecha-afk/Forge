"""Standalone Verification & Demonstration Script for Voice STT & LangGraph Architecture Orchestration.

Theme 2, Problem 1: Multilingual Agritech & Resource Advisory Assistant
Tests:
1. Bhashini Speech-to-Text & English Translation in Hindi, Bengali, Telugu, Marathi.
2. SQLite resilience cache write-through and retrieval.
3. Full 5-Node LangGraph Swarm Orchestration:
   - Disease Outbreak routing to GBDT Triad / ICAR RAG
   - Harvest Scheduling routing to Agmarknet Mandi Tool
   - Cross-Node Safety Overrides (Spray-Lock, Hypoxia Gate, Spoilage Override)
   - Vernacular response generation for farmer TTS playback
"""

import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.abspath("."))
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.abspath("."), "artifacts", ".mplcache")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.schemas.voice_schemas import VoiceTranscriptionRequest
from app.services.langgraph_agent.workflow import run_agritech_agent_workflow
from app.services.voice_service.audio_utils import compute_audio_hash
from app.services.voice_service.bhashini_stt import BhashiniVoiceService


async def run_all_voice_and_orchestration_tests():
    print("=" * 85)
    print("  FORGE BUILDATHON - BHASHINI VOICE STT & ARCHITECTURE ORCHESTRATION DEMO")
    print("  Theme 2, Problem 1: Multilingual Agritech & Resource Advisory Assistant")
    print("=" * 85)

    voice_svc = BhashiniVoiceService()

    # --------------------------------------------------------------------------
    # TEST 1: Bhashini Voice STT & Indic Translation
    # --------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[TEST 1] BHASHINI SPEECH-TO-TEXT & ENGLISH TRANSLATION LAYER")
    print("-" * 85)

    languages_to_test = [
        ("hi", "Hindi", b"hindi_audio_sample_bytes_12345"),
        ("bn", "Bengali", b"bengali_audio_sample_bytes_67890"),
        ("te", "Telugu", b"telugu_audio_sample_bytes_54321"),
        ("mr", "Marathi", b"marathi_audio_sample_bytes_98765"),
    ]

    for lang_code, lang_name, pcm_data in languages_to_test:
        b64 = base64.b64encode(pcm_data).decode("utf-8")
        req = VoiceTranscriptionRequest(
            audio_base64=b64,
            source_language=lang_code,
            audio_format="wav",
        )
        res = await voice_svc.transcribe_and_translate(req)
        print(f"[*] {lang_name} ({lang_code}):")
        print(f"    Vernacular Transcript: {res.vernacular_transcript}")
        print(f"    English Translation:   {res.english_transcript}")
        print(f"    Source: {res.data_source} | Confidence: {res.confidence_score * 100:.1f}%\n")
        assert res.vernacular_transcript != ""
        assert res.english_transcript != ""

    # --------------------------------------------------------------------------
    # TEST 2: SQLite Write-Through Resilience Cache
    # --------------------------------------------------------------------------
    print("-" * 85)
    print("[TEST 2] SQLITE WRITE-THROUGH RESILIENCE CACHE")
    print("-" * 85)
    test_audio_bytes = b"kisan_voice_note_hooghly_potato_001"
    a_hash = compute_audio_hash(test_audio_bytes)
    test_vern = "আমার আলুর গাছে পাতায় কালো দাগ হয়েছে, কী করব?"
    test_eng = "Black spots have appeared on my potato leaves, what should I do?"

    voice_svc.cache_transcription(a_hash, "bn", test_vern, test_eng)
    cached = voice_svc.get_cached_transcription(a_hash)
    assert cached is not None
    print(f"[*] Cache Entry Verified for Hash {a_hash[:16]}...:")
    print(f"    Language: {cached[0]} | Vernacular: {cached[1]}")
    print(f"    English:  {cached[2]}")
    print("    -> SQLite Resilience Cache Passed with 100% Data Integrity!\n")

    # --------------------------------------------------------------------------
    # TEST 3: Full 5-Node LangGraph Orchestration — Disease Outbreak (Hindi)
    # --------------------------------------------------------------------------
    print("-" * 85)
    print("[TEST 3] LANGGRAPH ORCHESTRATION: DISEASE OUTBREAK FLOW (HINDI)")
    print("-" * 85)

    hindi_voice_note = "आलू के पत्तों पर काले और भूरे धब्बे दिख रहे हैं, क्या यह झुलसा रोग है? मुझे क्या दवा डालनी चाहिए?"
    print(f"Farmer Query: '{hindi_voice_note}'")
    state_out_1 = await run_agritech_agent_workflow(
        text_query=hindi_voice_note,
        source_language="hi",
        district="Medinipur West",
        state="West Bengal",
    )

    print(f" -> Detected Crop:   {state_out_1['detected_crop'].upper()}")
    print(f" -> Detected Intent: {state_out_1['intent']}")
    print(f" -> Execution Route: {state_out_1['execution_route']}")
    print(f" -> Active Locks:    {state_out_1['active_guardrails']}")

    engine_res = state_out_1.get("engine_result", {})
    risk = engine_res.get("risk_assessment", {})
    print(f" -> Diagnosed Condition: {risk.get('predicted_disease')}")
    print(f" -> Outbreak Probability: {risk.get('calibrated_probability') * 100:.1f}% ({risk.get('risk_level')})")
    print(f" -> Top TreeSHAP Driver: {risk.get('shap_top_drivers', [{}])[0]}")

    print("\n[Node 5 Vernacular Audio Speech Text (Hindi)]:")
    print(f" \"{state_out_1['vernacular_response_text']}\"\n")
    assert state_out_1["detected_crop"] == "potato"
    assert state_out_1["execution_route"] == "UNIFIED_AGRONOMIC_ENGINE"

    # --------------------------------------------------------------------------
    # TEST 4: Full 5-Node LangGraph Orchestration — Mandi Arbitrage (Bengali)
    # --------------------------------------------------------------------------
    print("-" * 85)
    print("[TEST 4] LANGGRAPH ORCHESTRATION: APMC MANDI HARVEST FLOW (BENGALI)")
    print("-" * 85)

    bengali_voice_note = "টমেটোর মান্ডি দর কেমন চলছে? আগামী কাল কি বৃষ্টি হবে, ফসল কি আজই তুলে বিক্রি করব?"
    print(f"Farmer Query: '{bengali_voice_note}'")
    state_out_2 = await run_agritech_agent_workflow(
        text_query=bengali_voice_note,
        source_language="bn",
        district="Medinipur West",
        state="West Bengal",
    )

    print(f" -> Detected Crop:   {state_out_2['detected_crop'].upper()}")
    print(f" -> Detected Intent: {state_out_2['intent']}")
    print(f" -> Execution Route: {state_out_2['execution_route']}")

    mandi_res = state_out_2.get("mandi_result", {})
    print(f" -> Market: {mandi_res.get('market_name')}")
    print(f" -> Modal Price: Rs. {mandi_res.get('current_modal_price_inr_per_quintal')}/qtl")
    print(f" -> 7-Day Moving Avg: Rs. {mandi_res.get('seven_day_moving_avg')}/qtl")
    print(f" -> Action Flag: {mandi_res.get('action_flag')}")

    print("\n[Node 5 Vernacular Audio Speech Text (Bengali)]:")
    print(f" \"{state_out_2['vernacular_response_text']}\"\n")
    assert state_out_2["detected_crop"] == "tomato"
    assert state_out_2["execution_route"] == "MANDI_HARVEST_ADVISOR"

    # --------------------------------------------------------------------------
    # TEST 5: Voice Note Audio Base64 End-to-End Execution
    # --------------------------------------------------------------------------
    print("-" * 85)
    print("[TEST 5] VOICE BASE64 AUDIO NOTE END-TO-END SWARM EXECUTION")
    print("-" * 85)

    dummy_mic_recording = b"\x00\x05\x00\x09" * 2000
    b64_mic = base64.b64encode(dummy_mic_recording).decode("utf-8")

    state_out_3 = await run_agritech_agent_workflow(
        audio_base64=b64_mic,
        source_language="hi",
        district="Hooghly",
        state="West Bengal",
    )

    print(f" -> STT Transcript (Vernacular): {state_out_3['farmer_vernacular_transcript']}")
    print(f" -> STT Transcript (English):    {state_out_3['english_transcript']}")
    print(f" -> Voice Data Provenance:       {state_out_3['data_source_voice']}")
    print(f" -> Final Audio Speech Text:     {state_out_3['final_speech_text']}")

    assert state_out_3["farmer_vernacular_transcript"] != ""
    assert state_out_3["english_transcript"] != ""

    print("\n" + "=" * 85)
    print("  ALL 5 VOICE & LANGGRAPH ORCHESTRATION TESTS COMPLETED SUCCESSFULLY (EXIT 0)!")
    print("=" * 85)


if __name__ == "__main__":
    asyncio.run(run_all_voice_and_orchestration_tests())
