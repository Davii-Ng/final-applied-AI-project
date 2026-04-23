import os
from typing import Any, Dict, List, Optional

from src.agents.agent1_mood import analyze_mood
from src.agents.agent2_profile import parse_profile
from src.agents.agent3_setlist import curate_setlist
from src.agents.agent4_narrator import narrate_setlist


def run_pipeline(
    user_message: str,
    songs: List[Dict[str, Any]],
    k: int = 5,
    agent1_backend: str = "local",
    agent1_model: str = "gemini-2.0-flash",
    agent1_api_key: Optional[str] = None,
    agent4_backend: str = "gemini",
    agent4_api_key: Optional[str] = None,
    optional_context: Optional[Dict[str, Any]] = None,
    persona: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent1_payload = analyze_mood(
        user_message=user_message,
        optional_context=optional_context,
        backend=agent1_backend,
        model=agent1_model,
        api_key=agent1_api_key,
    )

    trace_id = agent1_payload.get("trace_id")

    agent2_payload = parse_profile(
        user_message=user_message,
        agent1_payload=agent1_payload,
        optional_context=optional_context,
        trace_id=trace_id,
    )

    agent3_payload = curate_setlist(
        agent2_payload=agent2_payload,
        songs=songs,
        k=k,
        trace_id=trace_id,
    )

    agent4_payload = narrate_setlist(
        agent3_payload=agent3_payload,
        persona=persona,
        trace_id=trace_id,
        backend=agent4_backend,
        api_key=agent4_api_key or os.getenv("GOOGLE_API_KEY"),
    )

    return {
        "trace_id": trace_id,
        "agent1": agent1_payload,
        "agent2": agent2_payload,
        "agent3": agent3_payload,
        "agent4": agent4_payload,
    }
