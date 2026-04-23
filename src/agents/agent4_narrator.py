from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


def narrate_setlist(
    agent3_payload: Dict[str, Any],
    persona: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_trace = trace_id or str(agent3_payload.get("trace_id", "")).strip() or "trace-missing"

    setlist = agent3_payload.get("setlist", [])
    explanations = agent3_payload.get("explanations", [])
    profile = agent3_payload.get("profile_echo", {})

    if not isinstance(setlist, list) or not setlist:
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": resolved_trace,
            "intro": "I could not build a full setlist yet, but I can try again with a clearer vibe prompt.",
            "track_transitions": [],
            "closing": "Give me one more detail like genre, mood, or energy and I will refine the mix.",
            "safety_notes": ["empty_setlist_fallback"],
        }

    style = (persona or {}).get("style", "friendly")
    mood = profile.get("favorite_mood", "balanced") if isinstance(profile, dict) else "balanced"
    genre = profile.get("favorite_genre", "mixed") if isinstance(profile, dict) else "mixed"

    intro = f"Tonight's mix leans {mood} with a {genre} center. I built this sequence to keep energy coherent without repeating too much."

    transitions: List[str] = []
    for idx, item in enumerate(setlist):
        title = item.get("title", "Unknown Track")
        artist = item.get("artist", "Unknown Artist")
        rank = item.get("rank", idx + 1)
        reason = explanations[idx] if idx < len(explanations) else "selected for overall fit"
        transitions.append(f"#{rank}: {title} by {artist} - {reason}.")

    if style == "concise":
        transitions = transitions[: max(1, min(3, len(transitions)))]

    closing = "If you want a second pass, tell me what to push more: energy, mood, or genre diversity."

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": resolved_trace,
        "intro": intro,
        "track_transitions": transitions,
        "closing": closing,
        "safety_notes": [],
    }


class DJNarrator:
    """Agent 4 that transforms ranked setlists into DJ narration text."""

    def narrate(
        self,
        agent3_payload: Dict[str, Any],
        persona: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return narrate_setlist(agent3_payload=agent3_payload, persona=persona, trace_id=trace_id)
