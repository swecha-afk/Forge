"""Minimal Streamlit client for testing the FastAPI agent workflow."""

import base64
import binascii
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
LANGUAGES = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Kannada": "kn",
    "Punjabi": "pa",
    "Gujarati": "gu",
    "English": "en",
}

st.set_page_config(page_title="Agritech Workflow Test", page_icon="🌱")
st.title("Agritech Workflow Test")
st.caption(f"Backend: {BACKEND_URL}")

with st.form("agent-query"):
    language_name = st.selectbox("Response language", options=list(LANGUAGES))
    district = st.text_input("District", value="Hooghly")
    state_name = st.text_input("State", value="West Bengal")
    text_query = st.text_area("Query", placeholder="Describe the crop issue or request")
    audio_file = st.file_uploader(
        "Or upload a voice recording",
        type=["wav", "mp3", "ogg", "webm", "flac", "m4a"],
    )
    submitted = st.form_submit_button("Run workflow", type="primary")

if submitted:
    st.session_state.pop("workflow_result", None)
    if not text_query.strip() and audio_file is None:
        st.error("Enter a query or upload an audio recording.")
    else:
        payload = {
            "text_query": text_query.strip() if audio_file is None else None,
            "source_language": LANGUAGES[language_name],
            "district": district.strip(),
            "state": state_name.strip(),
        }
        if audio_file is not None:
            payload["audio_base64"] = base64.b64encode(
                audio_file.getvalue()
            ).decode("ascii")
            payload["audio_format"] = audio_file.name.rsplit(".", 1)[-1].lower()

        with st.spinner("Running the agent workflow..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/agent/orchestrate",
                    json=payload,
                    timeout=180,
                )
                if not response.ok:
                    st.error(f"Workflow failed ({response.status_code}): {response.text}")
                else:
                    st.session_state["workflow_result"] = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
            except ValueError as exc:
                st.error(f"Backend returned invalid JSON: {exc}")

result = st.session_state.get("workflow_result")
if result:
    st.subheader("Advisory")
    st.write(result.get("vernacular_response_text", "No vernacular advisory returned."))
    st.write(result.get("executive_advisory_en", ""))
    if result.get("safety_guardrails_active"):
        st.warning("\n".join(result["safety_guardrails_active"]))

    st.caption(
        f"Route: {result.get('execution_route', '—')} · "
        f"Crop: {result.get('detected_crop', '—')} · "
        f"Intent: {result.get('intent', '—')} · "
        f"Voice input: {result.get('data_source_voice', '—')}"
    )
    st.caption(
        f"Intent classifier: {result.get('intent_classification_source', 'KEYWORD_FALLBACK')} · "
        f"Response generator: {result.get('farmer_response_source', 'TEMPLATE_FALLBACK')}"
    )
    if result.get("llm_error"):
        st.warning(f"Groq LLM fallback: {result['llm_error']}")

    tts_status = result.get("tts_status", "TEXT_ONLY")
    st.caption(f"Sarvam speech output: {tts_status}")
    audio_base64 = result.get("tts_audio_base64")
    if audio_base64:
        try:
            st.audio(
                base64.b64decode(audio_base64, validate=True),
                format=result.get("tts_mime_type", "audio/wav"),
            )
        except (binascii.Error, ValueError) as exc:
            st.error(f"Received invalid TTS audio: {exc}")
    elif tts_status == "TTS_UNAVAILABLE":
        st.warning(
            result.get("tts_error")
            or "Speech audio is unavailable; the text advisory is shown above."
        )

    with st.expander("Workflow details"):
        st.json(result)
