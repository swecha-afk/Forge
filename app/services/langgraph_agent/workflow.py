"""LangGraph StateGraph Workflow Orchestrator for Agritech Swarm Assistant."""

from typing import Any
from langgraph.graph import END, StateGraph

from app.services.langgraph_agent.nodes import (
    guardrail_verification_node,
    router_node,
    stt_node,
    synthesizer_node,
    tool_execution_node,
    tts_output_node,
)
from app.services.langgraph_agent.state import AgritechAgentState


def build_agritech_agent_workflow():
    """Compiles the LangGraph workflow through the final TTS output node."""
    graph = StateGraph(AgritechAgentState)

    # 1. Add Nodes
    graph.add_node("stt", stt_node)
    graph.add_node("router", router_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("guardrail_verification", guardrail_verification_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("tts_output", tts_output_node)

    # 2. Add Deterministic Transitions
    graph.set_entry_point("stt")
    graph.add_edge("stt", "router")
    graph.add_edge("router", "tool_execution")
    graph.add_edge("tool_execution", "guardrail_verification")
    graph.add_edge("guardrail_verification", "synthesizer")
    graph.add_edge("synthesizer", "tts_output")
    graph.add_edge("tts_output", END)

    return graph.compile()


# Singleton compiled graph
_compiled_workflow = None


def get_agent_orchestrator():
    """Returns singleton compiled LangGraph orchestrator."""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_agritech_agent_workflow()
    return _compiled_workflow


async def run_agritech_agent_workflow(
    audio_base64: str | None = None,
    text_query: str | None = None,
    source_language: str = "hi",
    district: str = "Medinipur West",
    state: str = "West Bengal",
    crop_override: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    audio_format: str = "wav",
) -> dict[str, Any]:
    """Top-level invocation helper for FastAPI and Streamlit UI."""
    orchestrator = get_agent_orchestrator()

    initial_state: AgritechAgentState = {
        "audio_base64": audio_base64,
        "audio_format": audio_format,
        "source_language": source_language,
        "farmer_vernacular_transcript": text_query or "",
        "english_transcript": text_query or "",
        "data_source_voice": "DIRECT_TEXT_INPUT",
        "detected_crop": crop_override or "",
        "district": district,
        "state": state,
        "latitude": latitude,
        "longitude": longitude,
        "active_guardrails": [],
        "spray_lock_active": False,
        "irrigation_lock_active": False,
        "spoilage_override_active": False,
        "tts_audio_base64": None,
        "tts_status": "TEXT_ONLY",
        "tts_mime_type": "audio/wav",
        "tts_error": None,
    }

    final_state = await orchestrator.ainvoke(initial_state)
    return final_state
