# Project Handoff: Agritech Swarm Completion Guide

This document explains the new components that are currently untracked, their responsibilities, and the work remaining to make the project demo-ready.

## 1. Current Project State

The repository has a working multi-component backend and frontend structure. Most completed systems are connected through FastAPI and a LangGraph-style five-node workflow.

The current state is:

- Core ML and RAG components existed or were extended from the base project.
- New untracked code adds voice STT, multilingual orchestration, Streamlit UI, Docker deployment, and verification tests.
- The project has a high-quality demo foundation, but it is not fully verified end to end.
- The local environment used during review did not have LangGraph installed, so the orchestration suite could not be collected. This does not mean the tests passed.
- A complete Docker build and live API test have not yet been confirmed.

The Git working tree is not clean. Do not overwrite or revert existing modified files until the changes are reviewed with `git diff`.

Useful commands:

```powershell
git status
git diff --stat
git diff -- <file-path>
```

## 2. Important Ownership Boundary

This handoff identifies work for the next member but does not imply that every rubric claim is already proven. Treat the following as verified implementation areas, subject to final testing:

- LangGraph-style state orchestration
- XGBoost risk prediction with SHAP
- FAISS/SQLite ICAR retrieval
- Weather and mandi SQLite fallback
- Voice STT and transcript caching
- Streamlit evidence console
- Docker configuration

The following features are incomplete or not fully demonstrated:

- Actual spoken TTS audio output
- Complete failure handling with no cached weather
- Real end-to-end Docker verification
- Full automated test execution
- Production-grade secret handling

## 3. New Untracked Components

### 3.1 LangGraph Orchestration

Files:

- `app/services/langgraph_agent/__init__.py`
- `app/services/langgraph_agent/state.py`
- `app/services/langgraph_agent/nodes.py`
- `app/services/langgraph_agent/workflow.py`

Purpose:

- Moves the application from separate API calls into one state-based workflow.
- Defines and executes the five-node orchestration pipeline.
- Keeps routing, tool results, safety flags, and final response text in one typed state object.

Node flow:

1. `stt_node`
   - Accepts base64 voice audio.
   - Calls the Bhashini voice service when audio is present.
   - Otherwise passes text directly into the workflow.

2. `router_node`
   - Detects crop from known keywords.
   - Selects one of the intents: disease, harvest/mandi, irrigation, or general.
   - Resolves district latitude and longitude from a registry.

3. `tool_execution_node`
   - Sends mandi/harvest queries to the mandi advisor.
   - Sends disease and irrigation queries to the unified ML/RAG engine.
   - Loads the calibrated triad model and falls back to the baseline XGBoost model.

4. `guardrail_verification_node`
   - Collects active spray-lock, irrigation-lock, spoilage, and confidence guardrails.
   - Converts engine flags into visible state properties.

5. `synthesizer_node`
   - Builds English advisory text.
   - Builds templated vernacular text for the selected language.

Main entry function:

```python
run_agritech_agent_workflow(...)
```

The implementation is located in `app/services/langgraph_agent/workflow.py`.

Existing limitation:

- Routing is currently keyword-based. It is deterministic and suitable as a fallback, but it is not an LLM intent planner.
- The workflow is a sequential LangGraph state graph. Parallel tool execution has not been implemented.

### 3.2 Voice STT, Translation, and Cache

Files:

- `app/services/voice_service/__init__.py`
- `app/services/voice_service/audio_utils.py`
- `app/services/voice_service/bhashini_stt.py`

Purpose:

- Decodes base64 audio.
- Computes an audio SHA-256 hash.
- Calls Bhashini for Indic ASR and English translation.
- Stores successful results in SQLite.
- Falls back to cached or pre-seeded transcript scenarios.
- Builds vernacular response text for the output layer.

The main class is `BhashiniVoiceService`.

Supported language codes include:

- Hindi: `hi`
- Bengali: `bn`
- Telugu: `te`
- Marathi: `mr`
- Tamil: `ta`
- Kannada: `kn`
- Punjabi: `pa`
- Gujarati: `gu`
- English: `en`

Important limitations:

- There is no actual Bhashini TTS call.
- `synthesize_vernacular_speech_text()` creates text only; it does not generate an audio file or base64 audio response.
- Unknown cached audio can fall back to a representative pre-seeded scenario. This is useful for demo resilience but should be clearly identified as fallback data.
- The selected audio format is not propagated correctly in one workflow path: orchestration currently initializes `audio_format` as `"wav"` regardless of the uploaded file. Update this in `workflow.py`, the workflow node, and the frontend upload flow.

### 3.3 Voice API Schemas

Files:

- `app/schemas/voice_schemas.py`
- `app/routers/voice_agent_router.py`

The schemas define immutable API contracts for:

- Voice transcription requests and responses.
- End-to-end orchestrated requests and responses.
- Supported language codes.
- Farmer transcript, English transcript, route, crop, guardrails, and evidence payloads.

Main endpoints:

- `POST /api/v1/agent/voice/transcribe`
- `POST /api/v1/agent/orchestrate`
- `GET /api/v1/agent/languages`

Use the existing response schema as the frontend contract. Avoid changing field names without updating Streamlit and tests.

The orchestration router currently catches workflow exceptions and returns HTTP 500. The remaining work should add safer, structured error responses for:

- Weather unavailable.
- Cache unavailable.
- RAG unavailable or low confidence.
- Audio decode failure.
- Model artifact unavailable.
- Bhashini live/API failure.

### 3.4 Streamlit Frontend

File:

- `app/frontend/ui.py`

Purpose:

- Provides the farmer-facing left pane.
- Provides the judge verification console on the right.
- Accepts text and uploaded voice recordings.
- Displays preloaded demo scenarios.
- Displays vernacular advisory text.
- Displays SHAP results, ICAR citations, mandi data, guardrail flags, weather JSON, and benchmark evidence.

The frontend currently calls the workflow directly as a Python module instead of using FastAPI over HTTP. This is acceptable for the single-container demo, but a final architecture should call `/api/v1/agent/orchestrate` through the backend service when frontend/backend separation is required.

Remaining frontend work:

- Add a visible speaker/audio playback component after real TTS is implemented.
- Show whether data came from live APIs, SQLite cache, or fallback/demo data.
- Preserve original audio format instead of always sending WAV metadata.
- Improve validation for uploaded audio size and duration.
- Add an explicit unavailable-data state instead of only a generic engine error.
- Add a loading state for each external API or cache operation.

### 3.5 Docker Deployment

Files:

- `Dockerfile`
- `Dockerfile.frontend`
- `docker-compose.yml`
- `.env.example`

Purpose:

- Containerizes the FastAPI backend.
- Containerizes the Streamlit frontend.
- Shares artifact and data directories as volumes.
- Adds backend health checking and service dependency ordering.

The backend Dockerfile separately installs FAISS and SentenceTransformers because they are commented out in `requirements.txt`.

Remaining deployment work:

- Build both Docker images from a clean environment.
- Run the complete compose stack with the frontend profile.
- Verify that `/health` returns healthy.
- Verify that a `/api/v1/agent/orchestrate` request works inside Docker networking.
- Verify that Streamlit can reach the backend.
- Verify artifact and SQLite persistence after container restart.
- Remove public-looking API key values from defaults and `.env.example`.
- Use a local `.env` file for secrets and do not commit `.env`.

### 3.6 Dependency and Test Configuration

Files:

- `requirements.txt`
- `pytest.ini`
- `benchmarks/test_voice_and_orchestration.py`
- `benchmarks/run_voice_orchestration_demo.py`
- `artifacts/.mplcache/`

Purpose:

- Declares runtime dependencies.
- Configures pytest discovery.
- Provides orchestration, voice, and cache verification tests.
- Provides a runnable voice orchestration benchmark/demo.

The test suite includes:

- Hindi STT fallback/live behavior.
- Bengali STT fallback/live behavior.
- Voice cache write/retrieval round trips.
- Hindi disease outbreak orchestration.
- Bengali mandi/harvest orchestration.
- Voice-input orchestration routing.

Generated `artifacts/.mplcache/` contents should not normally be committed. It is Matplotlib runtime cache data, not project source.

### 3.7 Environment Configuration

`.env.example` documents environment variables for:

- Bhashini user ID, API key, and pipeline ID.
- Data.gov.in API key.
- SQLite database locations.
- Safety thresholds.
- API timeout.

Do not use production credentials in committed source. The Bhashini credentials should remain empty in the example.

Verify that every setting named in `.env.example` is also read consistently by `app/core/config.py`.

## 4. Modified Files That Must Be Reviewed

These files were modified but are not all new files. The new member should inspect their diffs before making coordinated edits.

- `WALKTHROUGH.md`
- `app/core/config.py`
- `app/main.py`
- `app/routers/__init__.py`
- `app/schemas/__init__.py`
- `app/services/mandi_service/fetcher.py`
- `app/services/rag_engine/db_manager.py`
- `app/services/rag_engine/ingest_icar.py`
- `app/services/rag_engine/retriever.py`
- `app/services/risk_engine/visualizer.py`
- `app/services/weather_api.py`

The modified RAG and weather files are tightly connected to the fallback guarantees. Review them before changing retriever thresholds or weather cache behavior.

## 5. Remaining Work, Prioritized

### P0: Required Before the Final Demo

#### 5.1 Implement Real TTS

Add a Bhashini TTS method and expose the result through the workflow and API.

Recommended implementation steps:

1. Add a TTS pipeline call in `app/services/voice_service/bhashini_stt.py`.
2. Return audio bytes or base64 audio in the service response.
3. Add `audio_base64`, MIME type, or audio URL fields to a response schema if needed.
4. Add a Streamlit audio player for the vernacular response.
5. Preserve normal text fallback when TTS credentials or network calls fail.
6. Mark whether output came from live TTS or text-only fallback.

Files likely to change:

- `app/services/voice_service/bhashini_stt.py`
- `app/services/langgraph_agent/nodes.py`
- `app/services/langgraph_agent/state.py`
- `app/schemas/voice_schemas.py`
- `app/routers/voice_agent_router.py`
- `app/frontend/ui.py`

#### 5.2 Fix Audio Format Propagation

The orchestration helper should accept the uploaded audio format.

Current issue:

- `run_agritech_agent_workflow()` initializes the workflow state with `"wav"` as the default audio format.
- Uploaded MP3, OGG, FLAC, and M4A data may still be processed with incorrect metadata.

Required change:

- Infer or receive the format in Streamlit.
- Pass it through `AgentVoiceQueryRequest`.
- Pass it through `run_agritech_agent_workflow()`.
- Pass it to `VoiceTranscriptionRequest`.

#### 5.3 Complete Failure Handling

The disease/agronomic branch currently invokes weather and raises when neither live nor cache data is available.

Required behavior:

- Catch the weather failure at a specific node level.
- Return an explicit `DATA_UNAVAILABLE` result or equivalent state field.
- Do not invent a risk probability, RAG treatment, or SHAP explanation from absent telemetry.
- Keep cached weather clearly labeled as cached data.
- Prevent the frontend from showing a normal advisory when only fallback/demo data is being used.

Also verify fallback behavior for:

- Missing model artifacts.
- FAISS unavailable.
- SentenceTransformers unavailable.
- SQLite database unavailable.
- Bhashini unavailable.
- Mandi API unavailable.
- Malformed base64 audio.

#### 5.4 Verify Docker End to End

Run:

```powershell
docker compose build
docker compose --profile frontend up
```

Test:

- `http://localhost:8000/health`
- FastAPI documentation at `http://localhost:8000/docs`
- Streamlit at `http://localhost:8501`
- One text disease query.
- One mandi query.
- One audio upload query.
- One forced weather/API failure.
- Container restart and SQLite persistence.

Do not mark Docker complete until all images start and the backend/frontend communicate successfully.

#### 5.5 Install Dependencies and Run the Tests

Install project and test dependencies in a clean environment, then run:

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest pytest-asyncio
python -m pytest -q
```

For all optional Docker dependencies, the Dockerfiles already install FAISS and SentenceTransformers. Add pytest and pytest-asyncio to a dedicated development requirements file if this remains a local development requirement.

### P1: Important Stability Work

#### 5.6 Improve Intent and Entity Routing

Replace fragile substring routing where needed with:

- Normalized crop synonyms.
- Negative/basic general query handling.
- Explicit intent labels.
- A user clarification response when crop/location/intent is ambiguous.
- Optional LLM routing with deterministic tool allow-listing, if the team intentionally wants an LLM planner.

Currently supported crops:

- Potato
- Tomato
- Rice
- Wheat

Expanding crops requires:

- Trained risk model classes or mapping.
- Weather and disease feature compatibility.
- RAG subspaces for each crop/disease.
- Frontend language support if needed.

#### 5.7 Make Text Translation an Explicit Workflow Step

Direct vernacular text currently remains in the state as both the source and English transcript. The first implementation should invoke an existing translation service for non-English text, or clearly remove unsupported translation claims.

At minimum:

- Add an explicit text translation function.
- Set `source_language`.
- Populate the English transcript with translated text.
- Keep the original text in `farmer_vernacular_transcript`.
- Use the English transcript for routing.

#### 5.8 Add a Structured TTS/Fallback Contract

Avoid displaying `"TTS unavailable"` as an agronomic result. Add fields such as:

- `tts_status`
- `audio_base64`
- `audio_mime_type`
- `data_source_voice`
- `fallback_reason`

This is also useful for judges, because it makes resilience evidence visible.

#### 5.9 Add Cache Provenance

The UI and API should explicitly identify:

- `LIVE`
- `CACHE`
- `PRESEEDED_FALLBACK`
- `TEXT_INPUT`
- `TTS_TEXT_ONLY`

This prevents a cached or demo response from being presented as fresh live advice.

#### 5.10 Improve API Contracts

Replace generic HTTP 500 responses with status-rich payloads for partial or failed workflows:

- `200` for successful advice.
- `206` for partial success if data is unavailable but cached content is usable.
- `422` for invalid payload/audio format.
- `503` for required services or required weather/model dependencies unavailable.

Only use HTTP 200 when an actionable advisory is actually returned.

### P2: Quality and Presentation Work

#### 5.11 Expand Automated Testing

Add tests for:

- Invalid base64 and unsupported audio format.
- Bhashini timeout, HTTP error, and malformed response.
- Cached voice lookup.
- Live weather timeout.
- Missing weather cache.
- Mandi timeout/rate limit.
- Missing model artifact.
- Missing FAISS artifact.
- RAG subspace empty.
- RAG confidence gate.
- No spray lock, irrigation lock, and spoilage lock.
- Each guardrail combination.
- Docker startup and health endpoint.
- FastAPI orchestration response model.
- Streamlit rendering with text-only and audio fallback states.

#### 5.12 Remove Generated Cache Files

Remove `artifacts/.mplcache/` before committing source. Add cache directories to `.gitignore`.

Example entries:

```gitignore
.env
.env.*
!.env.example
__pycache__/
.pytest_cache/
*.pyc
artifacts/.mplcache/
*.joblib.tmp
```

#### 5.13 Update Documentation

Update `WALKTHROUGH.md` and add or verify:

- Architecture diagram.
- Five-node execution description.
- Environment setup.
- Docker setup.
- Live API verification.
- Cache/failure demo.
- Test commands.
- Exact supported languages and crops.
- Known limitations.
- Demo fallback instructions.

#### 5.14 Prepare Rubric Evidence

Before presenting:

- Record a complete working backend and frontend demo.
- Prepare pre-recorded voice input.
- Prepare screenshots for Docker health, SHAP, RAG citations, and safety locks.
- Pre-cache weather for the demo location.
- Pre-cache mandi response for the demo commodity.
- Test one live API success and one forced fallback.
- Record evidence of container startup and restart persistence.

## 6. Recommended Execution Order

1. Implement actual TTS.
2. Fix audio format propagation.
3. Fix unhandled weather/cache/model/RAG failures.
4. Move API keys and defaults to environment-only configuration.
5. Install the full dependencies and run the orchestration suite.
6. Build and run Docker Compose.
7. Test one disease scenario and one mandi scenario.
8. Add missing failure tests.
9. Remove generated cache and debug files.
10. Update the walkthrough and record the final demo.

## 7. Definition of Completion

The project should be considered demo-complete only when all of these are true:

- A text or valid voice query reaches the final workflow response successfully.
- Crop, intent, weather, risk, RAG, and guardrail state are present where applicable.
- SHAP and ICAR evidence are visible in the judge console.
- The uploaded language and returned vernacular language are consistent.
- Real TTS audio plays, or text-only fallback is clearly labeled.
- Weather, mandi, voice, and cached fallback behavior are separately verifiable.
- Docker Compose starts without manual file edits.
- Secrets are loaded from environment variables and are not committed.
- Automated tests covering the core workflow and critical failures pass.
- The final walkthrough contains commands, screenshots, architecture, and limitations.
