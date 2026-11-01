"""Verification suite for Groq voice processing and LangGraph orchestration.

Tests:
1. Groq Speech-to-Text & English Translation across Indic languages.
2. Write-through SQLite resilience cache persistence and failover.
3. End-to-end LangGraph Swarm Orchestration, including the final TTS node:
   - Node 1: Voice STT
   - Node 2: Entity & Intent Router
   - Node 3: Tool Execution (Agronomic Triad ML / Mandi Tool)
   - Node 4: Guardrail Verification
   - Node 5: Vernacular TTS Text Synthesis
4. Verification of cross-node deterministic safety overrides within LangGraph.
"""

import base64
import json

import httpx
import pytest
import pytest_asyncio
from streamlit.testing.v1 import AppTest

from app.schemas.voice_schemas import (
    VoiceTranscriptionRequest,
    VoiceTranscriptionResponse,
)
from app.schemas.mandi_schema import MandiQueryRequest
from app.services.langgraph_agent import workflow
from app.services.langgraph_agent import nodes as agent_nodes
from app.services.langgraph_agent.nodes import (
    router_node,
    stt_node,
    synthesizer_node,
    tts_output_node,
    tool_execution_node,
)
from app.services.langgraph_agent.workflow import run_agritech_agent_workflow
from app.services.groq_agronomy import (
    GroqAgronomyService,
    GroqAgronomyUnavailableError,
)
from app.services import groq_agronomy as groq_agronomy_module
from app.services.mandi_service import scheduler as mandi_scheduler
from app.services.weather_api import LiveWeatherService
from app.services.voice_service.audio_utils import compute_audio_hash
from app.services.voice_service.bhashini_stt import (
    BhashiniVoiceService,
    VoiceServiceUnavailableError,
)
from app.services.voice_service.sarvam_tts import (
    SarvamTTSService,
    SarvamTTSUnavailableError,
)


def mock_groq_agronomy(monkeypatch, *, intent, crop, response="Mocked farmer response"):
    async def classify(self, vernacular_query, english_query, crop_hint=""):
        return {"intent": intent, "crop": crop}

    async def generate(self, **kwargs):
        return response

    monkeypatch.setattr(GroqAgronomyService, "classify_intent", classify)
    monkeypatch.setattr(
        GroqAgronomyService, "generate_farmer_response", generate
    )


@pytest.mark.asyncio
async def test_groq_retries_retired_default_model(monkeypatch):
    requested_models = []
    original_async_client = httpx.AsyncClient

    async def handler(request):
        requested_models.append(json.loads(request.content)["model"])
        if len(requested_models) == 1:
            return httpx.Response(
                404, json={"error": {"message": "model is no longer available"}}
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"intent":"general"}'}}]},
        )

    def create_client(*, timeout):
        return original_async_client(
            transport=httpx.MockTransport(handler), timeout=timeout
        )

    monkeypatch.setattr(groq_agronomy_module.httpx, "AsyncClient", create_client)
    service = GroqAgronomyService()
    service.api_key = "test-key"
    service.model = "llama-3.3-70b-versatile"

    result = await service._chat(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=10,
        temperature=0,
    )

    assert result == '{"intent":"general"}'
    assert requested_models == [
        "llama-3.3-70b-versatile",
        "qwen/qwen3.8-27b",
    ]
    assert service.model == "qwen/qwen3.8-27b"


@pytest.mark.asyncio
async def test_groq_404_includes_model_and_provider_message(monkeypatch):
    original_async_client = httpx.AsyncClient

    async def handler(_request):
        return httpx.Response(
            404, json={"error": {"message": "model not found"}}
        )

    def create_client(*, timeout):
        return original_async_client(
            transport=httpx.MockTransport(handler), timeout=timeout
        )

    monkeypatch.setattr(groq_agronomy_module.httpx, "AsyncClient", create_client)
    service = GroqAgronomyService()
    service.api_key = "test-key"
    service.model = "custom/unknown-model"

    with pytest.raises(
        GroqAgronomyUnavailableError,
        match="HTTP 404 for model 'custom/unknown-model'.*model not found",
    ):
        await service._chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
            temperature=0,
        )


@pytest.fixture(scope="session")
def voice_service():
    """Instantiates BhashiniVoiceService with SQLite cache."""
    return BhashiniVoiceService()


def test_weather_telemetry_reports_live_or_cached_source(tmp_path):
    """Labels weather telemetry provenance without changing its model features."""
    service = LiveWeatherService(db_path=str(tmp_path / "weather.db"))
    hourly = {
        "temperature_2m": [21.0],
        "relative_humidity_2m": [75.0],
        "precipitation": [0.0],
        "wind_speed_10m": [8.0],
        "soil_moisture_0_to_7cm": [0.22],
        "soil_moisture_7_to_28cm": [0.25],
    }

    live = service._parse_telemetry(
        "potato", {"hourly": hourly}, used_cache=False
    )
    cached = service._parse_telemetry(
        "potato", {"hourly": hourly}, used_cache=True
    )

    assert live.data_source == "LIVE_OPEN_METEO"
    assert cached.data_source == "SQLITE_FALLBACK_CACHE"
    assert live.to_feature_vector() == cached.to_feature_vector()


@pytest.mark.asyncio
async def test_mandi_schedule_marks_sample_baseline_as_demo(monkeypatch):
    """Prevents sample mandi prices from being presented as cached/live data."""
    async def no_live_data(*args, **kwargs):
        return None

    monkeypatch.setattr(mandi_scheduler, "fetch_live_mandi_data", no_live_data)
    monkeypatch.setattr(
        mandi_scheduler, "get_cached_latest_and_avg", lambda *args: None
    )

    result = await mandi_scheduler.evaluate_harvest_schedule(
        MandiQueryRequest(
            commodity="tomato",
            district="Unknown District",
            state="Unknown State",
        )
    )

    assert result.data_source == "DEMO_BASELINE"
    assert result.arrival_date == "Cached Baseline"


@pytest.mark.asyncio
async def test_groq_voice_stt_hindi(tmp_path, monkeypatch):
    """Verifies Groq results are returned and cached for Hindi audio."""
    voice_service = BhashiniVoiceService(db_path=str(tmp_path / "voice_cache.db"))
    # Simulated audio payload: 1 second of dummy PCM bytes
    dummy_audio = b"\x00\x01\x00\x02" * 4000
    b64_audio = base64.b64encode(dummy_audio).decode("utf-8")

    async def fake_groq_pipeline(**kwargs):
        return "आलू में रोग के लक्षण हैं", "The potato shows signs of disease"

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(voice_service, "_call_groq_pipeline", fake_groq_pipeline)
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
    assert resp.data_source == "GROQ_LIVE_API"
    assert resp.confidence_score is None


@pytest.mark.asyncio
async def test_groq_voice_stt_bengali(tmp_path, monkeypatch):
    """Verifies Bengali transcripts remain in Bengali and include English translation."""
    voice_service = BhashiniVoiceService(db_path=str(tmp_path / "voice_cache.db"))
    dummy_audio = b"\x01\x02\x03\x04" * 4000
    b64_audio = base64.b64encode(dummy_audio).decode("utf-8")

    async def fake_groq_pipeline(**kwargs):
        return "আলুর পাতায় দাগ দেখা যাচ্ছে", "Spots are visible on the potato leaves"

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(voice_service, "_call_groq_pipeline", fake_groq_pipeline)
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
    assert resp.data_source == "GROQ_LIVE_API"


@pytest.mark.asyncio
async def test_stt_node_passes_uploaded_audio_format_to_groq(monkeypatch):
    async def fake_transcribe(self, request):
        assert request.audio_format == "m4a"
        assert request.source_language == "bn"
        return VoiceTranscriptionResponse(
            vernacular_transcript="আলুর পাতায় দাগ",
            english_transcript="Spots on potato leaves",
            source_language="bn",
            data_source="GROQ_LIVE_API",
        )

    monkeypatch.setattr(BhashiniVoiceService, "__init__", lambda self: None)
    monkeypatch.setattr(
        BhashiniVoiceService, "transcribe_and_translate", fake_transcribe
    )
    result = await stt_node(
        {
            "audio_base64": base64.b64encode(b"sample voice recording").decode("ascii"),
            "audio_format": "m4a",
            "source_language": "bn",
        }
    )

    assert result == {
        "farmer_vernacular_transcript": "আলুর পাতায় দাগ",
        "english_transcript": "Spots on potato leaves",
        "data_source_voice": "GROQ_LIVE_API",
    }


def test_uploaded_audio_transcript_is_written_to_query_bar(monkeypatch):
    transcript = "आलू के पत्तों पर भूरे धब्बे दिखाई दे रहे हैं"
    workflow_calls = []

    async def fake_workflow(**kwargs):
        workflow_calls.append(kwargs)
        return {
            "farmer_vernacular_transcript": transcript,
            "english_transcript": "Brown spots are visible on potato leaves",
            "data_source_voice": "GROQ_LIVE_API",
            "vernacular_response_text": "सलाह",
            "final_speech_text": "सलाह",
            "tts_status": "TTS_UNAVAILABLE",
            "tts_error": "Test response",
            "active_guardrails": [],
            "engine_result": None,
            "mandi_result": None,
            "execution_route": "UNIFIED_AGRONOMIC_ENGINE",
            "detected_crop": "potato",
            "intent": "disease_outbreak",
        }

    monkeypatch.setattr(
        workflow, "run_agritech_agent_workflow", fake_workflow
    )
    app = AppTest.from_file("../app/frontend/ui.py", default_timeout=45).run()
    app.radio[0].set_value("Upload audio").run()
    app.file_uploader[0].upload(
        "question.wav", b"sample uploaded audio", "audio/wav"
    ).run()
    app.button[0].click().run()

    assert not app.exception
    assert workflow_calls[0]["audio_format"] == "wav"
    assert app.text_area[0].value == transcript
    assert any(
        "English interpretation:" in item.value
        for item in app.caption
    )


@pytest.mark.asyncio
async def test_groq_pipeline_transcribes_and_translates_indic_audio(tmp_path, monkeypatch):
    service = BhashiniVoiceService(db_path=str(tmp_path / "voice_cache.db"))
    calls = []

    class StubResponse:
        def __init__(self, text):
            self.status_code = 200
            self.text = text

        def raise_for_status(self):
            return None

        def json(self):
            return {"text": self.text}

    class StubAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def post(self, url, headers, data, files):
            calls.append((url, headers, data, files))
            if url.endswith("/audio/transcriptions"):
                return StubResponse("आलू के पत्तों पर दाग हैं")
            return StubResponse("There are spots on the potato leaves")

    monkeypatch.setattr(
        "app.services.voice_service.bhashini_stt.httpx.AsyncClient",
        StubAsyncClient,
    )
    vernacular, english = await service._call_groq_pipeline(
        audio_bytes=b"audio bytes",
        source_lang="hi",
        audio_format="ogg",
        api_key="test-groq-key",
    )

    assert vernacular == "आलू के पत्तों पर दाग हैं"
    assert english == "There are spots on the potato leaves"
    assert len(calls) == 2
    transcription_call = next(
        call for call in calls if call[0].endswith("/audio/transcriptions")
    )
    translation_call = next(
        call for call in calls if call[0].endswith("/audio/translations")
    )
    assert transcription_call[1]["Authorization"] == "Bearer test-groq-key"
    assert transcription_call[2]["model"] == "whisper-large-v3"
    assert transcription_call[2]["language"] == "hi"
    assert transcription_call[3]["file"][0] == "recording.ogg"
    assert "language" not in translation_call[2]


@pytest.mark.asyncio
async def test_groq_english_audio_uses_transcription_only(tmp_path, monkeypatch):
    service = BhashiniVoiceService(db_path=str(tmp_path / "voice_cache.db"))
    calls = []

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "The potato leaves have spots"}

    class StubAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def post(self, url, **kwargs):
            calls.append(url)
            return StubResponse()

    monkeypatch.setattr(
        "app.services.voice_service.bhashini_stt.httpx.AsyncClient",
        StubAsyncClient,
    )
    vernacular, english = await service._call_groq_pipeline(
        audio_bytes=b"audio bytes",
        source_lang="en",
        audio_format="wav",
        api_key="test-groq-key",
    )

    assert vernacular == english == "The potato leaves have spots"
    assert calls == ["https://api.groq.com/openai/v1/audio/transcriptions"]


@pytest.mark.asyncio
async def test_uncached_audio_requires_groq_key(tmp_path, monkeypatch):
    service = BhashiniVoiceService(db_path=str(tmp_path / "voice_cache.db"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.services.voice_service.bhashini_stt.settings.GROQ_API_KEY", ""
    )
    request = VoiceTranscriptionRequest(
        audio_base64=base64.b64encode(b"unseen audio").decode("ascii"),
        source_language="hi",
    )

    with pytest.raises(VoiceServiceUnavailableError, match="GROQ_API_KEY"):
        await service.transcribe_and_translate(request)


@pytest.mark.asyncio
async def test_cached_exact_audio_transcript_is_used_without_groq(tmp_path, monkeypatch):
    service = BhashiniVoiceService(db_path=str(tmp_path / "voice_cache.db"))
    audio_bytes = b"previously transcribed audio"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.services.voice_service.bhashini_stt.settings.GROQ_API_KEY", ""
    )
    service.cache_transcription(
        compute_audio_hash(audio_bytes),
        "hi",
        "आलू के पत्तों पर दाग हैं",
        "There are spots on potato leaves",
    )
    request = VoiceTranscriptionRequest(
        audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
        source_language="hi",
    )

    response = await service.transcribe_and_translate(request)

    assert response.data_source == "RESILIENCE_CACHE_FAILOVER"
    assert response.vernacular_transcript == "आलू के पत्तों पर दाग हैं"
    assert response.english_transcript == "There are spots on potato leaves"


@pytest.mark.asyncio
async def test_voice_sqlite_cache_roundtrip(voice_service):
    """Verifies that Groq transcripts are stored and retrieved with full fidelity."""
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
async def test_langgraph_orchestrator_disease_outbreak_flow(monkeypatch):
    """Verifies full LangGraph orchestration on a Hindi disease outbreak query."""
    mock_groq_agronomy(
        monkeypatch, intent="disease_outbreak", crop="potato"
    )
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
    assert final_state["intent_classification_source"] == "GROQ_LLM"
    assert final_state["engine_result"] is not None

    engine_res = final_state["engine_result"]
    assert "risk_assessment" in engine_res
    assert "rag_guidance" in engine_res
    if engine_res["rag_guidance"]["chemical_synthesis_blocked"]:
        assert final_state["farmer_response_source"] == "SAFETY_TEMPLATE"
    else:
        assert final_state["farmer_response_source"] == "GROQ_RAG_GROUNDED"
    assert len(engine_res["risk_assessment"]["shap_top_drivers"]) == 3

    # Node 5 Vernacular synthesis output check
    assert final_state["vernacular_response_text"] != ""
    assert len(final_state["final_speech_text"]) > 20


@pytest.mark.asyncio
async def test_groq_classifier_validates_structured_result(monkeypatch):
    service = GroqAgronomyService()
    service.api_key = "test-key"

    async def valid_completion(**kwargs):
        return '{"intent":"irrigation_check","crop":"rice"}'

    monkeypatch.setattr(service, "_chat", valid_completion)
    result = await service.classify_intent(
        vernacular_query="ধানের জমিতে কী করব?",
        english_query="What should I do in my rice field?",
    )

    assert result == {"intent": "irrigation_check", "crop": "rice"}

    async def unsupported_completion(**kwargs):
        return '{"intent":"sell_land","crop":"rice"}'

    monkeypatch.setattr(service, "_chat", unsupported_completion)
    with pytest.raises(
        GroqAgronomyUnavailableError, match="unsupported intent"
    ):
        await service.classify_intent(
            vernacular_query="",
            english_query="",
        )


@pytest.mark.asyncio
async def test_groq_classifier_supports_crop_insurance_intent(monkeypatch):
    service = GroqAgronomyService()
    service.api_key = "test-key"
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return '{"intent":"crop_insurance","crop":"rice"}'

    monkeypatch.setattr(service, "_chat", completion)
    result = await service.classify_intent(
        vernacular_query="बाढ़ में धान की फसल बर्बाद हुई, PMFBY मुआवजा मिलेगा?",
        english_query="Flood damaged my rice crop. Can I claim PMFBY compensation?",
    )

    assert result == {"intent": "crop_insurance", "crop": "rice"}
    assert "crop_insurance" in captured["messages"][0]["content"]
    assert "not irrigation_check" in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_router_falls_back_when_groq_classification_fails(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise GroqAgronomyUnavailableError("Groq chat API returned HTTP 503.")

    monkeypatch.setattr(GroqAgronomyService, "classify_intent", unavailable)
    result = await router_node(
        {
            "farmer_vernacular_transcript": "potato leaves look unhealthy",
            "english_transcript": "potato leaves look unhealthy",
            "detected_crop": "",
            "district": "Hooghly",
        }
    )

    assert result["intent"] == "disease_outbreak"
    assert result["detected_crop"] == "potato"
    assert result["intent_classification_source"] == "KEYWORD_FALLBACK"
    assert "HTTP 503" in result["llm_error"]


@pytest.mark.asyncio
async def test_keyword_fallback_routes_pmfby_query_to_crop_insurance(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise GroqAgronomyUnavailableError("Groq chat API returned HTTP 503.")

    monkeypatch.setattr(GroqAgronomyService, "classify_intent", unavailable)
    result = await router_node(
        {
            "farmer_vernacular_transcript": (
                "बाढ़ में धान की फसल बर्बाद हुई, प्रधानमंत्री फसल बीमा योजना "
                "के तहत मुआवजा मिलेगा?"
            ),
            "english_transcript": "",
            "detected_crop": "",
            "district": "Hooghly",
        }
    )

    assert result["intent"] == "crop_insurance"
    assert result["detected_crop"] == "rice"
    assert result["intent_classification_source"] == "KEYWORD_FALLBACK"


@pytest.mark.asyncio
async def test_user_pmfby_flood_question_is_not_routed_to_irrigation(monkeypatch):
    async def misclassify(*args, **kwargs):
        return {"intent": "irrigation_check", "crop": "rice"}

    monkeypatch.setattr(GroqAgronomyService, "classify_intent", misclassify)
    result = await router_node(
        {
            "farmer_vernacular_transcript": (
                "कि पिछले हफ्ते आई बाढ़ में मेरी लगभग पचास प्रतिशत धान की फसल "
                "बर्बाद हो गई क्या मुझे प्रधानमंत्री फसल बीमायोजना के तहत "
                "मुआवजा मिल सकता है आवेदन करने की अंतिम तारीख और जरूरी दस्तावेज "
                "कौन-कौन से हैं"
            ),
            "english_transcript": "",
            "detected_crop": "",
            "district": "Hooghly",
        }
    )

    assert result["intent"] == "crop_insurance"
    assert result["detected_crop"] == "rice"
    assert result["intent_classification_source"] == "GROQ_LLM_WITH_INSURANCE_OVERRIDE"


def test_keyword_fallback_routes_non_symptom_agronomy_query_to_general():
    result = agent_nodes._keyword_classification(
        "धान की खेती के लिए कौन सी किस्म चुनूं?"
    )

    assert result == {"intent": "general", "crop": "rice"}


@pytest.mark.asyncio
async def test_explicit_pmfby_query_overrides_wrong_irrigation_classification(
    monkeypatch,
):
    async def misclassify(*args, **kwargs):
        return {"intent": "irrigation_check", "crop": "rice"}

    monkeypatch.setattr(GroqAgronomyService, "classify_intent", misclassify)
    result = await router_node(
        {
            "farmer_vernacular_transcript": (
                "धान की फसल बर्बाद हुई, प्रधानमंत्री फसल बीमा योजना में "
                "मुआवजा और आवेदन की अंतिम तारीख क्या है?"
            ),
            "english_transcript": (
                "My rice crop was destroyed. What is the PMFBY compensation "
                "and application deadline?"
            ),
            "detected_crop": "",
            "district": "Hooghly",
        }
    )

    assert result["intent"] == "crop_insurance"
    assert result["detected_crop"] == "rice"
    assert result["intent_classification_source"] == "GROQ_LLM_WITH_INSURANCE_OVERRIDE"
    assert agent_nodes._keyword_classification("धान का पौधा बीमार है")[
        "intent"
    ] == "disease_outbreak"


@pytest.mark.asyncio
async def test_crop_insurance_uses_dedicated_route_without_weather_engine():
    result = await tool_execution_node(
        {
            "intent": "crop_insurance",
            "detected_crop": "rice",
            "district": "Hooghly",
            "state": "West Bengal",
            "farmer_vernacular_transcript": "धान की फसल का PMFBY दावा कैसे करें?",
            "english_transcript": "How do I claim PMFBY for rice?",
        }
    )

    assert result["execution_route"] == "CROP_INSURANCE_GUIDANCE"
    assert result["insurance_guidance"]["crop"] == "rice"
    assert result["insurance_guidance"]["current_official_policy_evidence_available"] is False
    assert result["engine_result"] is None
    assert result["mandi_result"] is None


@pytest.mark.asyncio
async def test_general_query_uses_general_route_without_weather_engine():
    result = await tool_execution_node(
        {
            "intent": "general",
            "detected_crop": "rice",
            "district": "Hooghly",
            "state": "West Bengal",
            "farmer_vernacular_transcript": "धान की कौन सी किस्म लगाऊं?",
            "english_transcript": "Which rice variety should I plant?",
        }
    )

    assert result["execution_route"] == "GENERAL_AGRICULTURE_GUIDANCE"
    assert result["engine_result"] is None
    assert result["mandi_result"] is None


@pytest.mark.asyncio
async def test_groq_response_generation_uses_selected_language_and_evidence(
    monkeypatch,
):
    service = GroqAgronomyService()
    service.api_key = "test-key"
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return "जवाब"

    monkeypatch.setattr(service, "_chat", completion)
    answer = await service.generate_farmer_response(
        source_language="hi",
        vernacular_query="मेरी फसल में समस्या है",
        english_query="My crop has an issue",
        intent="disease_outbreak",
        crop="potato",
        evidence={"icar_guidance": {"citations": [{"source_doc_title": "ICAR"}]}},
    )

    assert answer == "जवाब"
    assert "workflow_evidence" in captured["messages"][1]["content"]
    assert "Hindi" in captured["messages"][0]["content"]
    assert "not instructions" in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_groq_policy_prompt_avoids_unverified_deadlines(monkeypatch):
    service = GroqAgronomyService()
    service.api_key = "test-key"
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return "जवाब"

    monkeypatch.setattr(service, "_chat", completion)
    await service.generate_farmer_response(
        source_language="hi",
        vernacular_query="PMFBY में बाढ़ के नुकसान पर दावा कैसे करें?",
        english_query="How to claim flood damage under PMFBY?",
        intent="crop_insurance",
        crop="rice",
        evidence={"insurance_guidance": {"current_official_policy_evidence_available": False}},
    )

    prompt = captured["messages"][0]["content"]
    assert "Do not state an exact deadline" in prompt
    assert "current official state notification" in prompt


@pytest.mark.asyncio
async def test_insurance_response_does_not_use_irrigation_lock_or_weather_evidence(
    monkeypatch,
):
    class StubVoiceService:
        def synthesize_vernacular_speech_text(self, *args, **kwargs):
            return "wrong generic template"

    captured = {}

    async def generate(self, **kwargs):
        captured.update(kwargs)
        return "यह PMFBY के बारे में जवाब है।"

    monkeypatch.setattr(
        agent_nodes, "BhashiniVoiceService", lambda: StubVoiceService()
    )
    monkeypatch.setattr(
        GroqAgronomyService, "generate_farmer_response", generate
    )
    result = await synthesizer_node(
        {
            "source_language": "hi",
            "detected_crop": "rice",
            "intent": "crop_insurance",
            "insurance_guidance": {
                "program": "PMFBY",
                "state": "West Bengal",
                "district": "Hooghly",
                "crop": "rice",
                "current_official_policy_evidence_available": False,
            },
            "weather_telemetry": {"soil_moisture_root_zone": 0.36},
            "active_guardrails": ["[CRITICAL_HOLD_IRRIGATION]"],
        }
    )

    assert result["farmer_response_source"] == "GROQ_POLICY_GUIDANCE"
    assert result["vernacular_response_text"] == "यह PMFBY के बारे में जवाब है।"
    assert captured["intent"] == "crop_insurance"
    assert captured["evidence"]["insurance_guidance"]["program"] == "PMFBY"
    assert captured["evidence"]["telemetry"] is None
    assert captured["evidence"]["safety_guardrails"]["hold_irrigation"] is False
    assert captured["evidence"]["active_guardrails"] == []


@pytest.mark.asyncio
async def test_insurance_response_falls_back_to_insurance_text(monkeypatch):
    class StubVoiceService:
        def synthesize_vernacular_speech_text(self, *args, **kwargs):
            pytest.fail("Insurance fallback must not use an agronomy template")

    async def unavailable(self, **kwargs):
        raise GroqAgronomyUnavailableError("Groq chat API returned HTTP 503.")

    monkeypatch.setattr(
        agent_nodes, "BhashiniVoiceService", lambda: StubVoiceService()
    )
    monkeypatch.setattr(
        GroqAgronomyService, "generate_farmer_response", unavailable
    )
    result = await synthesizer_node(
        {
            "source_language": "hi",
            "detected_crop": "rice",
            "intent": "crop_insurance",
            "insurance_guidance": {
                "program": "PMFBY",
                "current_official_policy_evidence_available": False,
            },
        }
    )

    assert result["farmer_response_source"] == "TEMPLATE_FALLBACK"
    assert "बीमा" in result["vernacular_response_text"]
    assert "मिट्टी में नमी" not in result["vernacular_response_text"]
    assert "यूरिया" not in result["vernacular_response_text"]
    assert "503" in result["llm_error"]


@pytest.mark.asyncio
async def test_irrigation_lock_does_not_override_disease_response(monkeypatch):
    class StubVoiceService:
        def synthesize_vernacular_speech_text(self, *args, **kwargs):
            return "disease template"

    async def generate(self, **kwargs):
        return "disease-specific response"

    monkeypatch.setattr(
        agent_nodes, "BhashiniVoiceService", lambda: StubVoiceService()
    )
    monkeypatch.setattr(
        GroqAgronomyService, "generate_farmer_response", generate
    )
    result = await synthesizer_node(
        {
            "source_language": "hi",
            "detected_crop": "rice",
            "intent": "disease_outbreak",
            "engine_result": {
                "executive_advisory": "disease advisory",
                "risk_assessment": {"predicted_disease": "blast"},
                "safety_guardrails": {"hold_irrigation": True},
                "rag_guidance": {
                    "chemical_synthesis_blocked": False,
                    "citations": [],
                },
            },
        }
    )

    assert result["farmer_response_source"] == "GROQ_RAG_GROUNDED"
    assert result["vernacular_response_text"] == "disease-specific response"


@pytest.mark.asyncio
async def test_synthesizer_keeps_blocked_rag_on_safety_template(monkeypatch):
    class StubVoiceService:
        def synthesize_vernacular_speech_text(self, *args, **kwargs):
            return "safe template"

    async def should_not_generate(*args, **kwargs):
        pytest.fail("Groq must not generate advice when the ICAR gate blocks it")

    monkeypatch.setattr(
        agent_nodes, "BhashiniVoiceService", lambda: StubVoiceService()
    )
    monkeypatch.setattr(
        GroqAgronomyService, "generate_farmer_response", should_not_generate
    )
    result = await synthesizer_node(
        {
            "source_language": "en",
            "detected_crop": "potato",
            "intent": "disease_outbreak",
            "engine_result": {
                "executive_advisory": "advisory",
                "risk_assessment": {"predicted_disease": "blight"},
                "safety_guardrails": {},
                "rag_guidance": {
                    "chemical_synthesis_blocked": True,
                    "citations": [],
                },
            },
        }
    )

    assert result["farmer_response_source"] == "SAFETY_TEMPLATE"
    assert "no chemical" in result["vernacular_response_text"].lower()


@pytest.mark.asyncio
async def test_synthesizer_passes_bounded_icar_evidence_to_groq(monkeypatch):
    class StubVoiceService:
        def synthesize_vernacular_speech_text(self, *args, **kwargs):
            return "template"

    captured = {}

    async def generate(self, **kwargs):
        captured.update(kwargs)
        return "grounded response"

    monkeypatch.setattr(
        agent_nodes, "BhashiniVoiceService", lambda: StubVoiceService()
    )
    monkeypatch.setattr(
        GroqAgronomyService, "generate_farmer_response", generate
    )
    citations = [
        {
            "source_doc_title": f"ICAR {index}",
            "active_ingredient": "verified ingredient",
            "raw_content": "x" * 1500,
            "private_field": "not passed",
        }
        for index in range(4)
    ]
    result = await synthesizer_node(
        {
            "source_language": "hi",
            "detected_crop": "potato",
            "intent": "disease_outbreak",
            "engine_result": {
                "executive_advisory": "advisory",
                "risk_assessment": {"predicted_disease": "blight"},
                "safety_guardrails": {},
                "rag_guidance": {
                    "status": "VERIFIED",
                    "chemical_synthesis_blocked": False,
                    "citations": citations,
                },
            },
        }
    )
    passed_citations = captured["evidence"]["icar_guidance"]["citations"]

    assert result["farmer_response_source"] == "GROQ_RAG_GROUNDED"
    assert result["vernacular_response_text"] == "grounded response"
    assert len(passed_citations) == 3
    assert len(passed_citations[0]["raw_content"]) == 1200
    assert "private_field" not in passed_citations[0]


@pytest.mark.asyncio
async def test_langgraph_orchestrator_harvest_mandi_flow(monkeypatch):
    """Verifies full LangGraph orchestration on a Bengali Mandi harvest query."""
    mock_groq_agronomy(
        monkeypatch, intent="harvest_scheduling", crop="tomato"
    )
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
async def test_langgraph_orchestrator_voice_input_flow(monkeypatch):
    """Verifies voice input transcribes through Groq before workflow routing."""
    mock_groq_agronomy(
        monkeypatch, intent="disease_outbreak", crop="potato"
    )
    dummy_audio = b"\x00\x02\x04\x06" * 3000
    b64_audio = base64.b64encode(dummy_audio).decode("utf-8")

    async def fake_groq_pipeline(**kwargs):
        return (
            "आलू के पत्तों पर धब्बे हैं",
            "There are spots on the potato leaves",
        )

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(
        BhashiniVoiceService,
        "_call_groq_pipeline",
        lambda self, **kwargs: fake_groq_pipeline(**kwargs),
    )

    async def fake_tool_execution(state):
        return {
            "engine_result": {
                "executive_advisory": "Test advisory",
                "risk_assessment": {
                    "predicted_disease": "leaf_blight",
                    "calibrated_probability": 0.4,
                },
                "rag_guidance": {
                    "chemical_synthesis_blocked": False,
                    "citations": [
                        {
                            "source_doc_title": "ICAR test",
                            "active_ingredient": "verified",
                            "dilution_dosage": "as cited",
                        }
                    ],
                },
                "safety_guardrails": {
                    "hold_spray": False,
                    "hold_irrigation": False,
                },
            },
            "mandi_result": None,
            "execution_route": "UNIFIED_AGRONOMIC_ENGINE",
            "weather_data_source": "TEST",
            "weather_telemetry": {},
        }

    async def fake_tts_output(state):
        return {
            "tts_audio_base64": None,
            "tts_status": "TEXT_ONLY",
            "tts_mime_type": "audio/wav",
            "tts_error": None,
        }

    monkeypatch.setattr(workflow, "tool_execution_node", fake_tool_execution)
    monkeypatch.setattr(workflow, "tts_output_node", fake_tts_output)
    monkeypatch.setattr(workflow, "_compiled_workflow", None)
    final_state = await run_agritech_agent_workflow(
        audio_base64=b64_audio,
        source_language="hi",
        district="Hooghly",
        state="West Bengal",
    )

    assert final_state["farmer_vernacular_transcript"] != ""
    assert final_state["english_transcript"] != ""
    assert final_state["data_source_voice"] in [
        "GROQ_LIVE_API",
        "RESILIENCE_CACHE_FAILOVER",
    ]
    assert final_state["execution_route"] in [
        "UNIFIED_AGRONOMIC_ENGINE",
        "MANDI_HARVEST_ADVISOR",
    ]


@pytest.mark.parametrize(
    ("target_language", "language_code"),
    [
        ("hi", "hi-IN"),
        ("bn", "bn-IN"),
        ("te", "te-IN"),
        ("mr", "mr-IN"),
        ("en", "en-IN"),
    ],
)
@pytest.mark.asyncio
async def test_sarvam_tts_returns_base64_audio_in_selected_language(
    monkeypatch, target_language, language_code
):
    audio_content = base64.b64encode(b"fake wav bytes").decode("ascii")
    calls = []

    class StubResponse:
        status_code = 200

        def json(self):
            midpoint = len(audio_content) // 2
            return {"audios": [audio_content[:midpoint], audio_content[midpoint:]]}

    class StubAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def post(self, url, json, headers):
            calls.append((url, json, headers))
            return StubResponse()

    monkeypatch.setattr(
        "app.services.voice_service.sarvam_tts.httpx.AsyncClient", StubAsyncClient
    )
    monkeypatch.setenv("SARVAM_API_KEY", "test-sarvam-key")
    monkeypatch.setattr(
        "app.services.voice_service.sarvam_tts.settings.SARVAM_TTS_MODEL",
        "bulbul:v3",
    )
    monkeypatch.setattr(
        "app.services.voice_service.sarvam_tts.settings.SARVAM_TTS_SPEAKER",
        "shubh",
    )
    result = await SarvamTTSService().text_to_speech(
        text="test advisory",
        target_language=target_language,
    )

    assert result == (audio_content, "audio/wav")
    assert len(calls) == 1
    assert calls[0][0] == "https://api.sarvam.ai/text-to-speech"
    assert calls[0][1]["text"] == "test advisory"
    assert calls[0][1]["language_code"] == language_code
    assert calls[0][1]["model"] == "bulbul:v3"
    assert calls[0][1]["speaker"] == "shubh"
    assert calls[0][2]["api-subscription-key"] == "test-sarvam-key"


@pytest.mark.asyncio
async def test_sarvam_tts_requires_api_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.services.voice_service.sarvam_tts.settings.SARVAM_API_KEY", ""
    )

    with pytest.raises(SarvamTTSUnavailableError, match="SARVAM_API_KEY"):
        await SarvamTTSService().text_to_speech("नमस्ते", target_language="hi")


@pytest.mark.asyncio
async def test_tts_output_node_returns_live_audio(monkeypatch):
    audio_content = base64.b64encode(b"spoken advisory").decode("ascii")

    async def fake_text_to_speech(self, text, target_language):
        assert text == "सलाह"
        assert target_language == "bn"
        return audio_content, "audio/wav"

    monkeypatch.setattr(SarvamTTSService, "text_to_speech", fake_text_to_speech)
    result = await tts_output_node(
        {"final_speech_text": "सलाह", "source_language": "bn"}
    )

    assert result == {
        "tts_audio_base64": audio_content,
        "tts_status": "TTS_LIVE",
        "tts_mime_type": "audio/wav",
        "tts_error": None,
    }


@pytest.mark.asyncio
async def test_tts_output_node_exposes_sarvam_failure(monkeypatch):
    async def unavailable(self, text, target_language):
        raise SarvamTTSUnavailableError("Sarvam TTS returned HTTP 401.")

    monkeypatch.setattr(SarvamTTSService, "text_to_speech", unavailable)
    result = await tts_output_node(
        {"final_speech_text": "सलाह", "source_language": "hi"}
    )

    assert result["tts_audio_base64"] is None
    assert result["tts_status"] == "TTS_UNAVAILABLE"
    assert result["tts_error"] == "Sarvam TTS returned HTTP 401."


@pytest.mark.asyncio
async def test_tts_output_node_does_not_speak_hindi_fallback_as_another_language(
    monkeypatch,
):
    async def should_not_synthesize(self, text, target_language):
        pytest.fail("TTS must not be called for a language without a response template")

    monkeypatch.setattr(SarvamTTSService, "text_to_speech", should_not_synthesize)
    result = await tts_output_node(
        {"final_speech_text": "सलाह", "source_language": "ta"}
    )

    assert result["tts_status"] == "TTS_UNAVAILABLE"
    assert "wrong language" in result["tts_error"]


@pytest.mark.asyncio
async def test_workflow_preserves_uploaded_audio_format(monkeypatch):
    class StubOrchestrator:
        async def ainvoke(self, state):
            self.state = state
            return state

    orchestrator = StubOrchestrator()
    monkeypatch.setattr(workflow, "get_agent_orchestrator", lambda: orchestrator)

    await run_agritech_agent_workflow(audio_base64="audio", audio_format="mp3")

    assert orchestrator.state["audio_format"] == "mp3"
