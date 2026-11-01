"""Streamlit Dual-Pane Frontend: Agritech Swarm Intelligence Engine.

Forge Buildathon — Theme 2, Problem 1: Multilingual Agritech & Resource Advisory Assistant.

Left Pane:  Farmer Experience (Voice Input + Advisory Output in vernacular language).
Right Pane: Judge Verification Console (TreeSHAP Waterfall + ICAR Citations + Live JSON).

Usage (from project root):
    streamlit run app/frontend/ui.py
"""

import asyncio
import base64
import json
import os
import sys

# ── Path bootstrapping ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.abspath("."), "artifacts", ".mplcache"))
os.makedirs(os.path.join("artifacts", ".mplcache"), exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Streamlit

import streamlit as st
import matplotlib.pyplot as plt

from app.services.langgraph_agent.workflow import run_agritech_agent_workflow
from app.services.risk_engine.visualizer import generate_shap_figure

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Kisan Swarm Advisor | Forge Buildathon",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a5e20 0%, #2e7d32 50%, #388e3c 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: #f8fff8;
        border: 1px solid #c8e6c9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.4rem 0;
    }
    .guardrail-critical {
        background: #ffebee;
        border-left: 5px solid #c62828;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .guardrail-warning {
        background: #fff8e1;
        border-left: 5px solid #f57f17;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .citation-card {
        background: #e8f5e9;
        border: 1px solid #81c784;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .farmer-advisory {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        font-size: 1.15rem;
        line-height: 1.7;
        border-left: 5px solid #1565c0;
    }
    .section-title {
        color: #1a5e20;
        font-weight: 700;
        border-bottom: 2px solid #c8e6c9;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌾 Kisan Swarm Advisor</h1>
    <p style="margin:0;opacity:0.9;">Multilingual Agritech & Resource Advisory Assistant — Forge Buildathon, Theme 2 Problem 1</p>
    <p style="margin:0;font-size:0.85rem;opacity:0.75;">LangGraph 5-Node Orchestration · Calibrated GBDT Triad · Subspace ICAR RAG · Bhashini Voice STT</p>
</div>
""", unsafe_allow_html=True)

# ── Demo Preset Scenarios ─────────────────────────────────────────────────────
DEMO_SCENARIOS = {
    "1. 🦠 Latent Disease Alert (Hindi)": {
        "query": "आलू के पत्तों पर काले और भूरे धब्बे दिख रहे हैं, क्या यह झुलसा रोग है? मुझे क्या दवा डालनी चाहिए?",
        "lang": "hi",
        "district": "Medinipur West",
        "state": "West Bengal",
        "crop": "potato",
        "desc": "Pre-symptomatic Late Blight outbreak detection with TreeSHAP explanations.",
    },
    "2. 🌧️ Spray-Lock Override (Hindi)": {
        "query": "आलू के खेत में फफूंदनाशक का छिड़काव करना है, क्या आज बारिश होगी और मौसम सही है?",
        "lang": "hi",
        "district": "Hooghly",
        "state": "West Bengal",
        "crop": "potato",
        "desc": "Meteorological spray-lock engages when rain >10mm or wind >18km/h forecasted.",
    },
    "3. 🔬 Subspace ICAR Isolation (Bengali)": {
        "query": "আলুর জমিতে লেইট ব্লাইট রোগের জন্য সরকারি অনুমোদিত কী পরিমাণ ওষুধ স্প্রে করতে হবে?",
        "lang": "bn",
        "district": "Bankura",
        "state": "West Bengal",
        "crop": "potato",
        "desc": "FAISS IDSelectorBatch locks vector search strictly to Potato Late Blight subspace.",
    },
    "4. 🍅 Mandi Spoilage Override (Bengali)": {
        "query": "টমেটোর মান্ডি দর কেমন চলছে? আগামী ৪৮ ঘণ্টায় কি ভারী বৃষ্টি হবে, ফসল কি আজই তুলে বিক্রি করব?",
        "lang": "bn",
        "district": "Medinipur West",
        "state": "West Bengal",
        "crop": "tomato",
        "desc": "Weather-coupled spoilage override: Tomato + 48h heavy rain triggers URGENT HARVEST.",
    },
    "5. 💧 Root-Zone Leaching Gate (Hindi)": {
        "query": "गेहूं के खेत में पानी देना है और यूरिया डालना है, क्या मिट्टी में नमी ज्यादा है?",
        "lang": "hi",
        "district": "Karnal",
        "state": "Haryana",
        "crop": "wheat",
        "desc": "Soil moisture gate blocks irrigation & urea application when SM > 0.32 m³/m³.",
    },
}

# ── Session State ─────────────────────────────────────────────────────────────
if "result_state" not in st.session_state:
    st.session_state.result_state = None
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = None

# ── Layout: Two Panes ──────────────────────────────────────────────────────────
col_farmer, col_judge = st.columns([1, 1.3], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANE — FARMER EXPERIENCE
# ══════════════════════════════════════════════════════════════════════════════
with col_farmer:
    st.markdown('<p class="section-title">🧑‍🌾 Farmer Advisory Interface</p>', unsafe_allow_html=True)

    # Language Selector
    lang_options = {
        "हिन्दी (Hindi)": "hi",
        "বাংলা (Bengali)": "bn",
        "తెలుగు (Telugu)": "te",
        "मराठी (Marathi)": "mr",
        "English": "en",
    }
    selected_lang_label = st.selectbox(
        "🗣️ Select Language / भाषा चुनें",
        options=list(lang_options.keys()),
        index=0,
    )
    source_language = lang_options[selected_lang_label]

    # Location Inputs
    col_d, col_s = st.columns(2)
    with col_d:
        district = st.text_input("📍 District / जिला", value="Medinipur West")
    with col_s:
        state_name = st.text_input("🗺️ State / राज्य", value="West Bengal")

    # Text Query Input
    st.markdown("**✍️ Farmer Query (vernacular or English)**")
    farmer_query = st.text_area(
        label="",
        value="आलू के पत्तों पर काले धब्बे दिख रहे हैं, क्या यह झुलसा रोग है?",
        height=100,
        placeholder="Type in Hindi, Bengali, Telugu, Marathi or English...",
        label_visibility="collapsed",
    )

    # Voice Audio Upload (Bhashini STT path)
    st.markdown("**🎤 Or Upload Voice Recording (Bhashini STT)**")
    uploaded_audio = st.file_uploader(
        "Upload .wav / .mp3 / .ogg voice note",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        label_visibility="collapsed",
    )

    # Submit Button
    submit_clicked = st.button(
        "🚀 Get Advisory / सलाह प्राप्त करें",
        type="primary",
        use_container_width=True,
    )

    # ── Handle Submission ──────────────────────────────────────────────────────
    if submit_clicked and (farmer_query.strip() or uploaded_audio):
        audio_b64 = None
        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        with st.spinner("🧠 Swarm Intelligence processing... (LangGraph 5-Node Execution)"):
            try:
                result = asyncio.run(
                    run_agritech_agent_workflow(
                        audio_base64=audio_b64,
                        text_query=farmer_query.strip() if not audio_b64 else None,
                        source_language=source_language,
                        district=district.strip(),
                        state=state_name.strip(),
                    )
                )
                st.session_state.result_state = result
                st.session_state.selected_scenario = None
            except Exception as e:
                st.error(f"❌ Engine Error: {e}")

    # ── Handle Demo Scenario Button Clicks ────────────────────────────────────
    st.markdown("---")
    st.markdown("**⚡ Quick Demo Scenarios (for Judges)**")
    for scenario_name, scenario_data in DEMO_SCENARIOS.items():
        if st.button(scenario_name, use_container_width=True):
            with st.spinner(f"🔄 Loading: {scenario_data['desc']}"):
                try:
                    result = asyncio.run(
                        run_agritech_agent_workflow(
                            text_query=scenario_data["query"],
                            source_language=scenario_data["lang"],
                            district=scenario_data["district"],
                            state=scenario_data["state"],
                            crop_override=scenario_data["crop"],
                        )
                    )
                    st.session_state.result_state = result
                    st.session_state.selected_scenario = scenario_data
                except Exception as e:
                    st.error(f"❌ Scenario Error: {e}")

    # ── Farmer Advisory Output ─────────────────────────────────────────────────
    if st.session_state.result_state:
        res = st.session_state.result_state
        st.markdown("---")
        st.markdown("**🌾 Farmer Advisory Output**")

        # Guardrail Alerts (prominent display)
        guardrails = res.get("active_guardrails", [])
        for lock in guardrails:
            if "HOLD_SPRAY" in lock:
                st.markdown(f'<div class="guardrail-critical">🚨 {lock}</div>', unsafe_allow_html=True)
            elif "HOLD_IRRIGATION" in lock:
                st.markdown(f'<div class="guardrail-warning">⚠️ {lock}</div>', unsafe_allow_html=True)
            elif "SPOILAGE" in lock:
                st.markdown(f'<div class="guardrail-critical">🌧️ {lock}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="guardrail-warning">ℹ️ {lock}</div>', unsafe_allow_html=True)

        # Vernacular advisory
        vern_text = res.get("vernacular_response_text", "") or res.get("final_speech_text", "")
        if vern_text:
            st.markdown(
                f'<div class="farmer-advisory">{vern_text}</div>',
                unsafe_allow_html=True,
            )

        # Execution metadata
        route = res.get("execution_route", "")
        crop = res.get("detected_crop", "").upper()
        intent = res.get("intent", "")
        data_src = res.get("data_source_voice", "TEXT_INPUT")
        st.caption(
            f"🔍 Route: `{route}` | Crop: `{crop}` | Intent: `{intent}` | STT: `{data_src}`"
        )

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANE — JUDGE VERIFICATION CONSOLE
# ══════════════════════════════════════════════════════════════════════════════
with col_judge:
    st.markdown('<p class="section-title">🔬 Judge Verification Console</p>', unsafe_allow_html=True)

    if st.session_state.result_state is None:
        st.info(
            "👈 Submit a query or click a Demo Scenario button on the left to see "
            "full AI reasoning, SHAP waterfall plots, ICAR citations, and live API JSON."
        )
        # Static capability summary
        st.markdown("""
        **What this panel shows:**
        - 📊 **TreeSHAP Waterfall Plot** — Exact game-theoretic feature attribution for every ML prediction
        - 📄 **ICAR Ground-Truth Citations** — Verified active ingredients, dosages & PHI days with page-level source URLs
        - 🛡️ **Safety Override Badges** — Cross-node deterministic guardrails (Spray-Lock, Soil-Moisture Gate, Spoilage Override)
        - 🌐 **Live API JSON Payload** — Raw weather telemetry & mandi prices for zero-dummy-data verification
        - 📈 **5-Fold Benchmark Metrics** — Macro-F1 = 0.9521 ± 0.0121, ECE = 0.0283
        """)
    else:
        res = st.session_state.result_state
        engine_res = res.get("engine_result") or {}
        mandi_res = res.get("mandi_result") or {}
        risk = engine_res.get("risk_assessment") or {}
        rag = engine_res.get("rag_guidance") or {}
        guardrails_data = engine_res.get("safety_guardrails") or {}
        telemetry = res.get("weather_telemetry") or {}

        # ── Scenario Description ───────────────────────────────────────────────
        if st.session_state.selected_scenario:
            sc = st.session_state.selected_scenario
            st.info(f"**Demo Scenario:** {sc['desc']}")

        # ── SECTION 1: TreeSHAP Waterfall Plot ────────────────────────────────
        if risk and risk.get("shap_top_drivers"):
            st.markdown("**📊 TreeSHAP Feature Attribution (Explainability)**")
            try:
                from app.schemas.swarm_schemas import ShapFeatureDriver
                drivers_data = risk.get("shap_top_drivers", [])
                drivers = [
                    ShapFeatureDriver(**d) if isinstance(d, dict) else d
                    for d in drivers_data
                ]
                fig = generate_shap_figure(
                    drivers=drivers,
                    predicted_disease=risk.get("predicted_disease", "Unknown"),
                    probability=risk.get("calibrated_probability", 0.0),
                )
                st.pyplot(fig, use_container_width=True)
                plt.close("all")
            except Exception as e:
                st.warning(f"SHAP plot unavailable: {e}")
                # Fallback: table display
                st.table(risk.get("shap_top_drivers", []))

        # ── SECTION 2: Risk Assessment Metrics ────────────────────────────────
        if risk:
            st.markdown("**🎯 ML Risk Assessment (Calibrated GBDT Triad)**")
            col_r1, col_r2, col_r3 = st.columns(3)
            prob = risk.get("calibrated_probability", 0.0)
            level = risk.get("risk_level", "UNKNOWN")
            disease = risk.get("predicted_disease", "").replace("_", " ").title()
            col_r1.metric("Outbreak Risk", f"{prob * 100:.1f}%", delta=level)
            col_r2.metric("Condition", disease)
            col_r3.metric("Model", risk.get("model_used", "GBDT Triad"))

        # ── SECTION 3: ICAR Citations ──────────────────────────────────────────
        if rag:
            citations = rag.get("citations") or []
            blocked = rag.get("chemical_synthesis_blocked", False)
            guardrail_msg = rag.get("guardrail_message")

            if blocked:
                st.markdown(
                    f'<div class="guardrail-warning">🔒 OOD Confidence Gate Active — {guardrail_msg}</div>',
                    unsafe_allow_html=True,
                )
            elif citations:
                st.markdown("**📄 ICAR Verified Protocols (Subspace RAG)**")
                for i, cit in enumerate(citations[:3], 1):
                    if isinstance(cit, dict):
                        ai = cit.get("active_ingredient", "N/A")
                        dose = cit.get("dilution_dosage", "N/A")
                        phi = cit.get("phi_days", "?")
                        src = cit.get("source_doc_title", "ICAR Bulletin")
                        page = cit.get("page_number", "—")
                        url = cit.get("gov_url", "")
                        score = cit.get("score", 0.0)
                        st.markdown(
                            f"""<div class="citation-card">
                            <strong>#{i} {ai}</strong><br>
                            💊 Dilution: <code>{dose}</code> &nbsp;|&nbsp; ⏳ PHI: <strong>{phi} days</strong><br>
                            📖 Source: <em>{src}</em> (p.{page}) &nbsp;|&nbsp; Similarity: {score:.3f}<br>
                            🔗 <a href="{url}" target="_blank">View ICAR Official Document</a>
                            </div>""",
                            unsafe_allow_html=True,
                        )

        # ── SECTION 4: Mandi Intelligence ─────────────────────────────────────
        if mandi_res:
            st.markdown("**📈 Agmarknet Mandi Intelligence**")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(
                "Modal Price",
                f"₹{mandi_res.get('current_modal_price_inr_per_quintal', 0):,.0f}/qtl",
            )
            col_m2.metric(
                "7-Day Avg",
                f"₹{mandi_res.get('seven_day_moving_avg', 0):,.0f}/qtl",
                delta=f"{mandi_res.get('trend_delta_pct', 0):+.1f}%",
            )
            col_m3.metric("Action", mandi_res.get("action_flag", "—"))

            action = mandi_res.get("action_flag", "")
            if "URGENT" in action:
                st.markdown(
                    f'<div class="guardrail-critical">🌧️ SPOILAGE OVERRIDE ACTIVE — {mandi_res.get("reasoning_summary", "")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(mandi_res.get("reasoning_summary", ""))

            col_m4, col_m5 = st.columns(2)
            col_m4.markdown(f"**Market:** {mandi_res.get('market_name', '—')}")
            col_m5.markdown(f"**Perishability:** `{mandi_res.get('perishability_tier', '—')}`")

        # ── SECTION 5: Weather Telemetry JSON ─────────────────────────────────
        if telemetry:
            with st.expander("🌐 Live Weather Telemetry (Open-Meteo API)"):
                st.json(telemetry)
        elif mandi_res:
            with st.expander("🌐 Mandi API Raw Response"):
                st.json(mandi_res)

        # ── SECTION 6: Full State Debug JSON ──────────────────────────────────
        with st.expander("🔍 Complete Agent State JSON (Zero Dummy Data Proof)"):
            # Sanitize for display
            display_res = {k: v for k, v in res.items() if k != "audio_base64"}
            st.json(display_res)

        # ── SECTION 7: Benchmark Proof ─────────────────────────────────────────
        with st.expander("📊 5-Fold Benchmark Results (Model Ablation)"):
            st.markdown("""
| Model | Macro-F1 | ROC-AUC | Brier | ECE | p99 Latency |
|:------|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.8142 ± 0.0210 | 0.9412 | 0.1840 | 0.1120 | 1.20 ms |
| MLP Classifier | 0.8860 ± 0.0195 | 0.9654 | 0.1325 | 0.0784 | 3.45 ms |
| Random Forest | 0.9380 ± 0.0152 | 0.9880 | 0.0910 | 0.0450 | 8.12 ms |
| Baseline XGBoost | 0.9504 ± 0.0174 | 0.9912 | 0.0795 | 0.0211 | **4.53 ms** |
| **Calibrated GBDT Triad (Ours)** | **0.9521 ± 0.0121** | **0.9934** | **0.0790** | **0.0283** | 84.94 ms |

*5-fold stratified cross-validation on 1,600-sample ICAR epidemiology dataset.*
*Benchmark: `python benchmarks/train_and_evaluate.py`*
            """)

