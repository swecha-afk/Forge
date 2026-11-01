"""Nodes for LangGraph Agritech Swarm Orchestrator.

Implements the LangGraph workflow through the final TTS node:
Node 1: Audio / Speech Input & Groq Whisper STT
Node 2: Agronomic Entity & Intent Router
Node 3: Intelligence Swarm & Tool Dispatcher
Node 4: Cross-Node Deterministic Guardrail Verification
Node 5: Vernacular Response Synthesis
Node 6: Sarvam TTS Audio Output
"""

import logging
import re
from typing import Any

from app.schemas.swarm_schemas import SUPPORTED_CROPS, WeatherTelemetryInput
from app.schemas.voice_schemas import VoiceTranscriptionRequest
from app.services.groq_agronomy import (
    GroqAgronomyService,
    GroqAgronomyUnavailableError,
)
from app.services.langgraph_agent.state import AgritechAgentState
from app.services.mandi_service.fetcher import normalize_commodity
from app.services.mandi_service.tool import mandi_harvest_advisor_tool
from app.services.rag_engine.retriever import DeterministicICARRAG
from app.services.risk_engine.base import get_predictor
from app.services.risk_engine.pipeline import UnifiedAgronomicEngine
from app.services.voice_service.bhashini_stt import (
    BhashiniVoiceService,
    VERNACULAR_SYNTHESIS_PATTERNS,
)
from app.services.voice_service.sarvam_tts import (
    SarvamTTSService,
    SarvamTTSUnavailableError,
)
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

CROP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "potato": ("potato", "aloo", "alu", "tuber", "आलू", "আলু", "బంగాళాదుంప", "बटाटा"),
    "tomato": ("tomato", "tamatar", "solanum", "टमाटर", "টমেটো", "టమాటా", "टोमॅटो"),
    "rice": ("rice", "paddy", "dhan", "chawal", "धान", "चावल", "ধান", "চাল", "వరి", "బియ్యం", "भात"),
    "wheat": ("wheat", "gehu", "kanak", "गेहूं", "गहूँ", "গম", "గోధుమ", "गहू"),
}

HARVEST_KEYWORDS = (
    "mandi", "price", "rate", "sell", "harvest", "market", "bhav", "daam", "spoil",
    "मंडी", "बाजार", "भाव", "दाम", "बेच", "कटाई",
    "মান্ডি", "মন্ডি", "বাজার", "দর", "দাম", "বিক্রি", "তুলে", "কাট",
    "మార్కెట్", "మండి", "ధర", "రేటు", "అమ్మ", "కోత",
    "मंडी", "बाजार", "दर", "भाव", "विक्री", "काढणी",
)

IRRIGATION_KEYWORDS = (
    "water", "irrigate", "irrigation", "soil", "moisture", "urea",
    "fertilizer", "pump", "hypoxia",
    "पानी", "सिंचाई", "मिट्टी", "नमी", "खाद", "उर्वरक",
    "জল", "সেচ", "মাটি", "আর্দ্রতা", "সার",
    "నీరు", "నీటిపారుదల", "నేల", "తేమ", "ఎరువు",
    "पाणी", "सिंचन", "माती", "ओलावा", "खत",
)

DISEASE_KEYWORDS = (
    "disease", "symptom", "blight", "fungus", "fungal", "pest", "insect",
    "unhealthy", "sick", "yellow", "wilt", "rot", "spots", "spot on",
    "रोग", "बीमारी", "बीमार", "झुलसा", "धब्ब", "पीला", "मुरझा", "सड़न", "कीट", "फफूंद",
    "রোগ", "দাগ", "পোকা", "తెగులు", "మచ్చ", "పురుగు", "रोग", "डाग",
)

CROP_INSURANCE_KEYWORDS = (
    "pmfby",
    "crop insurance",
    "insurance claim",
    "crop-loss compensation",
    "compensation under",
    "प्रधानमंत्री फसल बीमा",
    "फसल बीमा",
    "बीमा योजना",
    "बीमायोजना",
    "मुआवजा",
    "मुआवज़ा",
    "क्षतिपूर्ति",
    "ফসল বীমা",
    "ফসল বিমা",
    "ক্ষতিপূরণ",
    "పంట బీమా",
    "పరిహారం",
    "पीक विमा",
    "नुकसानभरपाई",
)


def _is_crop_insurance_query(query: str) -> bool:
    normalized_query = query.lower()
    if any(keyword in normalized_query for keyword in CROP_INSURANCE_KEYWORDS):
        return True
    return bool(
        re.search(r"(?<!\w)(?:बीमा|বীমা|বিমা|భీమా)(?!\w)", normalized_query)
    )


# ------------------------------------------------------------------------------
# NODE 1: Audio / Speech Input & Groq Whisper STT
# ------------------------------------------------------------------------------
async def stt_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 1: Uses Groq to transcribe audio and translate it into English."""
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
def _keyword_classification(query: str) -> dict[str, str]:
    """Conservative local routing fallback for unavailable Groq classification."""
    normalized_query = query.lower()
    crop = next(
        (
            crop_name
            for crop_name, keywords in CROP_KEYWORDS.items()
            if any(keyword in normalized_query for keyword in keywords)
        ),
        "unknown",
    )
    if _is_crop_insurance_query(normalized_query):
        intent = "crop_insurance"
    elif any(keyword in normalized_query for keyword in HARVEST_KEYWORDS):
        intent = "harvest_scheduling"
    elif any(keyword in normalized_query for keyword in IRRIGATION_KEYWORDS):
        intent = "irrigation_check"
    elif any(keyword in normalized_query for keyword in DISEASE_KEYWORDS):
        intent = "disease_outbreak"
    else:
        intent = "general"
    return {"intent": intent, "crop": crop}


async def router_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 2: Classifies farmer intent and crop with Groq before tool dispatch."""
    vernacular_query = state.get("farmer_vernacular_transcript", "")
    english_query = state.get("english_transcript", "") or vernacular_query
    crop_hint = (state.get("detected_crop") or "").strip().lower()
    llm_error = None
    combined_query = f"{vernacular_query}\n{english_query}"

    try:
        classification = await GroqAgronomyService().classify_intent(
            vernacular_query=vernacular_query,
            english_query=english_query,
            crop_hint=crop_hint,
        )
        intent_source = "GROQ_LLM"
    except GroqAgronomyUnavailableError as error:
        classification = _keyword_classification(combined_query)
        intent_source = "KEYWORD_FALLBACK"
        llm_error = str(error)

    intent = classification["intent"]
    if _is_crop_insurance_query(combined_query):
        intent = "crop_insurance"
        if intent_source == "GROQ_LLM":
            intent_source = "GROQ_LLM_WITH_INSURANCE_OVERRIDE"
    crop = (
        crop_hint
        if crop_hint in SUPPORTED_CROPS
        else classification["crop"]
    )
    if crop not in SUPPORTED_CROPS:
        fallback_crop = _keyword_classification(combined_query)["crop"]
        if fallback_crop in SUPPORTED_CROPS:
            crop = fallback_crop
        elif intent in {"crop_insurance", "general"}:
            crop = "unknown"
        else:
            crop = "potato"

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
        "intent_classification_source": intent_source,
        "llm_error": llm_error,
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
    vernacular_query = state.get("farmer_vernacular_transcript", "")
    english_query = state.get("english_transcript", "")
    user_query = "\n".join(
        dict.fromkeys(query for query in (english_query, vernacular_query) if query)
    ) or "What is the recommended treatment and safety protocol?"

    if intent == "crop_insurance":
        return {
            "insurance_guidance": {
                "program": "PMFBY",
                "state": state_name,
                "district": district,
                "crop": crop if crop in SUPPORTED_CROPS else None,
                "current_official_policy_evidence_available": False,
            },
            "general_query_context": None,
            "engine_result": None,
            "mandi_result": None,
            "execution_route": "CROP_INSURANCE_GUIDANCE",
        }

    if intent == "general":
        return {
            "insurance_guidance": None,
            "general_query_context": {
                "query": user_query,
                "state": state_name,
                "district": district,
            },
            "engine_result": None,
            "mandi_result": None,
            "execution_route": "GENERAL_AGRICULTURE_GUIDANCE",
        }

    if intent == "harvest_scheduling":
        # Fetch upcoming 48h rain from weather API for Spoilage Override check
        weather_svc = LiveWeatherService()
        weather_data_source = "WEATHER_UNAVAILABLE"
        try:
            weather_data = weather_svc.fetch_live_telemetry(
                crop_name=crop, latitude=lat, longitude=lon
            )
            rain_48h = weather_data.rain_next_24h_mm * 1.5
            weather_data_source = weather_data.data_source or "UNKNOWN"
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
            "weather_data_source": weather_data_source,
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
        "weather_data_source": weather_telemetry.data_source or "UNKNOWN",
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
async def synthesizer_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 5: Grounds a farmer-language answer in tool output and retrieved ICAR evidence."""
    src_lang = state.get("source_language", "hi")
    crop = state.get("detected_crop", "potato").title()
    voice_service = BhashiniVoiceService()

    engine_res = state.get("engine_result")
    mandi_res = state.get("mandi_result")
    insurance_guidance = state.get("insurance_guidance")
    general_query_context = state.get("general_query_context")
    intent = state.get("intent", "general")
    guardrails: dict[str, Any] = {}
    rag: dict[str, Any] = {}

    if insurance_guidance:
        exec_en = (
            "No current official PMFBY notification or policy evidence is loaded. "
            "Enrolment deadlines and claim procedures vary by state, district, crop, "
            "and season; confirm them with the current official notification."
        )
        vern_text = {
            "hi": (
                "यह फसल बीमा/प्रधानमंत्री फसल बीमा योजना का सवाल है, सिंचाई का नहीं। "
                "अभी इस workflow में आपके राज्य, जिले, धान के मौसम और बीमा पॉलिसी की "
                "ताज़ा आधिकारिक अधिसूचना उपलब्ध नहीं है, इसलिए मैं अंतिम तारीख, दस्तावेज़ "
                "या मुआवज़े की पात्रता पक्के तौर पर नहीं बता सकता। नुकसान की सूचना अपने "
                "बीमाकर्ता और स्थानीय कृषि कार्यालय को तुरंत दें; सही प्रक्रिया वर्तमान "
                "राज्य अधिसूचना या आधिकारिक PMFBY पोर्टल से जाँचें।"
            ),
            "bn": (
                "এটি ফসল বিমা/PMFBY-সংক্রান্ত প্রশ্ন, সেচের নয়। আপনার রাজ্য, জেলা, "
                "ফসলের মরসুম এবং বিমা পলিসির বর্তমান সরকারি বিজ্ঞপ্তি এখানে নেই; তাই "
                "শেষ তারিখ, নথি বা ক্ষতিপূরণের যোগ্যতা নিশ্চিত করে বলা যাবে না। ক্ষতির "
                "কথা দ্রুত বিমাকারী ও স্থানীয় কৃষি দপ্তরকে জানান এবং বর্তমান সরকারি "
                "বিজ্ঞপ্তি বা PMFBY পোর্টালে নিয়ম যাচাই করুন।"
            ),
            "te": (
                "ఇది పంట బీమా/PMFBY గురించి; నీటిపారుదల గురించి కాదు. మీ రాష్ట్రం, "
                "జిల్లా, పంట సీజన్ మరియు బీమా పాలసీకి సంబంధించిన తాజా అధికారిక "
                "ప్రకటన ఇక్కడ లేదు. కాబట్టి చివరి తేదీ, పత్రాలు లేదా పరిహార అర్హతను "
                "ఖచ్చితంగా చెప్పలేను. నష్టాన్ని వెంటనే బీమా సంస్థకు, స్థానిక వ్యవసాయ "
                "కార్యాలయానికి తెలియజేసి అధికారిక ప్రకటనలో నియమాలు నిర్ధారించండి."
            ),
            "mr": (
                "हा प्रश्न पीक विमा/PMFBY बद्दल आहे, सिंचनाबद्दल नाही. तुमचे राज्य, "
                "जिल्हा, हंगाम आणि विमा पॉलिसीची सध्याची अधिकृत अधिसूचना येथे उपलब्ध "
                "नसल्याने अंतिम तारीख, कागदपत्रे किंवा भरपाईची पात्रता निश्चित सांगता "
                "येणार नाही. नुकसानीची माहिती त्वरित विमा कंपनी आणि स्थानिक कृषी "
                "कार्यालयाला द्या; सध्याच्या अधिकृत अधिसूचनेत नियम तपासा."
            ),
            "en": (
                "This is a crop-insurance/PMFBY question, not an irrigation question. "
                "The current official notification and your policy details are not "
                "available here, so I cannot verify the deadline, documents, or "
                "compensation eligibility. Report the loss promptly to your insurer "
                "and local agriculture office, and check the current official state "
                "notification or PMFBY portal."
            ),
        }.get(
            src_lang,
            "This is a crop-insurance question, not an irrigation question. The current "
            "official notification is unavailable here, so I cannot verify deadlines, "
            "documents, or eligibility. Report the loss promptly to your insurer and "
            "local agriculture office and check the current official notification.",
        )
    elif general_query_context:
        exec_en = "No specialized crop, insurance, mandi, or weather evidence was requested."
        vern_text = state.get("farmer_vernacular_transcript", "")
    elif mandi_res:
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
    else:
        # Otherwise synthesize from UnifiedAgronomicEngine result
        exec_en = engine_res.get("executive_advisory", "") if engine_res else ""
        guardrails = engine_res.get("safety_guardrails", {}) if engine_res else {}
        risk = engine_res.get("risk_assessment", {}) if engine_res else {}
        rag = engine_res.get("rag_guidance", {}) if engine_res else {}
        telemetry = state.get("weather_telemetry", {})

        if intent == "disease_outbreak" and guardrails.get("hold_spray"):
            vern_text = voice_service.synthesize_vernacular_speech_text(
                src_lang,
                "spray_lock",
                {
                    "rain": telemetry.get("rain_next_24h_mm", 12.0),
                    "wind": telemetry.get("wind_speed_kmh", 20.0),
                },
            )
        elif intent == "irrigation_check" and guardrails.get("hold_irrigation"):
            vern_text = voice_service.synthesize_vernacular_speech_text(
                src_lang,
                "irrigation_lock",
                {
                    "sm": int(telemetry.get("soil_moisture_root_zone", 0.35) * 100),
                },
            )
        elif intent == "disease_outbreak" and rag.get("chemical_synthesis_blocked"):
            vern_text = {
                "hi": (
                    "मौसम और फसल के आधार पर जोखिम का केवल प्रयोगात्मक अनुमान मिला है, "
                    "यह पक्का रोग-निदान नहीं है। उपलब्ध ICAR स्रोत इस मामले के लिए "
                    "प्रमाणित उपचार की पुष्टि नहीं करते, इसलिए दवा और मात्रा नहीं बताई जा रही।"
                ),
                "bn": (
                    "আবহাওয়া ও ফসলের ভিত্তিতে এটি একটি পরীক্ষামূলক ঝুঁকি-অনুমান, "
                    "নিশ্চিত রোগ-নির্ণয় নয়। উপলব্ধ ICAR সূত্রে এই সমস্যার জন্য "
                    "যাচাইকৃত চিকিৎসা নেই, তাই ওষুধ বা মাত্রা বলা হচ্ছে না।"
                ),
                "te": (
                    "వాతావరణం మరియు పంట ఆధారంగా ఇది ప్రయోగాత్మక ప్రమాద అంచనా మాత్రమే, "
                    "నిర్ధారిత రోగ నిర్ధారణ కాదు. అందుబాటులో ఉన్న ICAR ఆధారాలు "
                    "ధృవీకరించిన చికిత్సను చూపడం లేదు; మందు లేదా మోతాదు సూచించము."
                ),
                "mr": (
                    "हवामान आणि पिकावर आधारित हा केवळ प्रायोगिक जोखीम-अंदाज आहे, "
                    "निश्चित रोगनिदान नाही. उपलब्ध ICAR स्रोत या समस्येसाठी "
                    "पडताळलेला उपचार देत नाहीत, म्हणून औषध किंवा मात्रा सुचवत नाही."
                ),
                "en": (
                    "This is an experimental crop-and-weather risk estimate, not a "
                    "confirmed diagnosis. The retrieved ICAR sources do not verify a "
                    "treatment for this case, so no chemical or dosage is recommended."
                ),
            }.get(
                src_lang,
                "The retrieved ICAR sources do not verify a treatment for this case, "
                "so no chemical or dosage is recommended.",
            )
        else:
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
                    "chemical": top_cit.get("active_ingredient", ""),
                    "dose": top_cit.get("dilution_dosage", ""),
                    "phi": top_cit.get("phi_days", ""),
                },
            )

    risk_assessment = engine_res.get("risk_assessment", {}) if engine_res else {}
    rag_guidance = engine_res.get("rag_guidance", {}) if engine_res else {}
    applicable_guardrails = state.get("active_guardrails", [])
    if intent != "irrigation_check":
        applicable_guardrails = [
            item
            for item in applicable_guardrails
            if "HOLD_IRRIGATION" not in item
        ]
    if intent != "disease_outbreak":
        applicable_guardrails = [
            item
            for item in applicable_guardrails
            if "HOLD_SPRAY" not in item and "CONFIDENCE_GATE" not in item
        ]
    if intent != "harvest_scheduling":
        applicable_guardrails = [
            item for item in applicable_guardrails if "SPOILAGE_OVERRIDE" not in item
        ]
    evidence = {
        "executive_advisory": exec_en,
        "risk_assessment": {
            "predicted_disease": risk_assessment.get("predicted_disease"),
            "calibrated_probability": risk_assessment.get(
                "calibrated_probability"
            ),
            "risk_level": risk_assessment.get("risk_level"),
        },
        "icar_guidance": {
            "status": rag_guidance.get("status"),
            "chemical_synthesis_blocked": rag_guidance.get(
                "chemical_synthesis_blocked"
            ),
            "guardrail_message": rag_guidance.get("guardrail_message"),
            "citations": [
                {
                    "source_doc_title": citation.get("source_doc_title"),
                    "page_number": citation.get("page_number"),
                    "gov_url": citation.get("gov_url"),
                    "crop_name": citation.get("crop_name"),
                    "disease_or_pest": citation.get("disease_or_pest"),
                    "active_ingredient": citation.get("active_ingredient"),
                    "dilution_dosage": citation.get("dilution_dosage"),
                    "phi_days": citation.get("phi_days"),
                    "score": citation.get("score"),
                    "raw_content": str(citation.get("raw_content", ""))[:1200],
                }
                for citation in rag_guidance.get("citations", [])[:3]
            ],
        },
        "safety_guardrails": {
            "hold_spray": (
                intent == "disease_outbreak" and guardrails.get("hold_spray", False)
            ),
            "hold_irrigation": (
                intent == "irrigation_check"
                and guardrails.get("hold_irrigation", False)
            ),
        },
        "telemetry": (
            state.get("weather_telemetry")
            if intent in {"disease_outbreak", "irrigation_check"}
            else None
        ),
        "weather_source": state.get("weather_data_source"),
        "mandi_result": mandi_res,
        "insurance_guidance": insurance_guidance,
        "general_query_context": general_query_context,
        "active_guardrails": applicable_guardrails,
    }
    farmer_response_source = "TEMPLATE_FALLBACK"
    llm_error = state.get("llm_error")
    safety_template_required = bool(
        (intent == "disease_outbreak" and guardrails.get("hold_spray"))
        or (intent == "irrigation_check" and guardrails.get("hold_irrigation"))
        or (
            intent == "disease_outbreak"
            and rag.get("chemical_synthesis_blocked")
        )
        or (
            intent == "harvest_scheduling"
            and mandi_res
            and mandi_res.get("action_flag") == "URGENT_HARVEST_WEATHER_RISK"
        )
    )
    if safety_template_required:
        farmer_response_source = "SAFETY_TEMPLATE"
    else:
        try:
            vern_text = await GroqAgronomyService().generate_farmer_response(
                source_language=src_lang,
                vernacular_query=state.get("farmer_vernacular_transcript", ""),
                english_query=state.get("english_transcript", ""),
                intent=state.get("intent", "general"),
                crop=crop.lower(),
                evidence=evidence,
            )
            if intent == "crop_insurance":
                farmer_response_source = "GROQ_POLICY_GUIDANCE"
            elif intent == "general":
                farmer_response_source = "GROQ_GENERAL_GUIDANCE"
            else:
                farmer_response_source = "GROQ_RAG_GROUNDED"
        except GroqAgronomyUnavailableError as error:
            logger.warning("Grounded Groq response unavailable: %s", error)
            llm_error = "; ".join(
                part for part in (llm_error, str(error)) if part
            )

    return {
        "executive_advisory_en": exec_en,
        "vernacular_response_text": vern_text,
        "final_speech_text": vern_text,
        "farmer_response_source": farmer_response_source,
        "llm_error": llm_error,
    }


# ------------------------------------------------------------------------------
# NODE 6: TTS Audio Output — Sarvam Text-to-Speech
# ------------------------------------------------------------------------------
async def tts_output_node(state: AgritechAgentState) -> dict[str, Any]:
    """Node 6: Speaks the synthesized response using Sarvam in the selected language."""
    speech_text = state.get("final_speech_text", "")
    src_lang = state.get("source_language", "hi")

    if not speech_text or not speech_text.strip():
        logger.warning("TTS node received empty speech text — returning TEXT_ONLY.")
        return {
            "tts_audio_base64": None,
            "tts_status": "TEXT_ONLY",
            "tts_mime_type": "audio/wav",
            "tts_error": None,
        }

    if src_lang not in VERNACULAR_SYNTHESIS_PATTERNS:
        message = (
            f"A written advisory template is not available for language '{src_lang}'; "
            "audio was not generated to avoid speaking the wrong language."
        )
        logger.warning(message)
        return {
            "tts_audio_base64": None,
            "tts_status": "TTS_UNAVAILABLE",
            "tts_mime_type": "audio/wav",
            "tts_error": message,
        }

    try:
        audio_b64, mime_type = await SarvamTTSService().text_to_speech(
            text=speech_text, target_language=src_lang
        )
    except SarvamTTSUnavailableError as error:
        logger.warning("Sarvam TTS unavailable: %s", error)
        return {
            "tts_audio_base64": None,
            "tts_status": "TTS_UNAVAILABLE",
            "tts_mime_type": "audio/wav",
            "tts_error": str(error),
        }

    return {
        "tts_audio_base64": audio_b64,
        "tts_status": "TTS_LIVE",
        "tts_mime_type": mime_type,
        "tts_error": None,
    }
